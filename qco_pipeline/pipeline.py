"""
End-to-end inference pipeline: raw QASM in -> semantically-verified balanced
QASM out, wiring together every module built above.

    raw circuit
        |
        v
   [Stage 1 model]  ->  Checksum 1  --fail-->  classical stage1_fallback
        |  pass
        v
   [Stage 2 model]  ->  Checksum 2  --fail-->  classical stage2_fallback
        |  pass
        v
   balanced circuit (semantically verified, k <= n qubits)

Every decision (model output accepted vs. fallback triggered, and why) is
appended to the diagnostics log — this is the audit trail referenced in the
skill's "Fallback Handling & Diagnostics" output section.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from qco_pipeline.fallback.classical_fallback import (
    DiagnosticEvent,
    DiagnosticsLog,
    stage1_fallback,
    stage2_fallback,
)
from qco_pipeline.graph.circuit_graph import qasm_to_graph
from qco_pipeline.models.apply_stage1 import apply_stage1_actions
from qco_pipeline.models.apply_stage2 import apply_stage2_mask
from qco_pipeline.models.stage1_horizontal import Stage1HorizontalModel
from qco_pipeline.models.stage2_vertical import Stage2VerticalModel
from qco_pipeline.verification.checksum1 import run_checksum1
from qco_pipeline.verification.checksum2 import run_checksum2

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    balanced_qasm: str
    n_qubits_in: int
    n_qubits_out: int
    stage1_used_fallback: bool
    stage2_used_fallback: bool
    checksum1_fidelity: float
    checksum2_fidelity: float | None  # None if Stage 2 pruned nothing (checksum 2 skipped, checksum 1 suffices)


class QuantumCircuitOptimizer:
    def __init__(self, stage1_ckpt: Path, stage2_ckpt: Path, diagnostics_path: Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.stage1 = Stage1HorizontalModel().to(self.device)
        self.stage1.load_state_dict(torch.load(stage1_ckpt, map_location=self.device)["model_state"])
        self.stage1.eval()

        self.stage2 = Stage2VerticalModel().to(self.device)
        self.stage2.load_state_dict(torch.load(stage2_ckpt, map_location=self.device)["model_state"])
        self.stage2.eval()

        self.diagnostics = DiagnosticsLog(diagnostics_path)

    def optimize(self, raw_qasm: str, source_name: str = "<inline>") -> OptimizationResult:
        # ---- Stage 1 -------------------------------------------------
        graph = qasm_to_graph(raw_qasm)
        graph.data = graph.data.to(self.device)

        with torch.no_grad():
            action_logits, predicted_angle = self.stage1(graph.data)

        stage1_used_fallback = False
        try:
            stage1_qasm = apply_stage1_actions(raw_qasm, action_logits.cpu(), predicted_angle.cpu())
            checksum1 = run_checksum1(raw_qasm, stage1_qasm)
        except Exception as exc:  # noqa: BLE001
            checksum1 = None
            logger.warning("Stage 1 reconstruction raised %s — routing to fallback", exc)

        if checksum1 is None or not checksum1.passed:
            stage1_used_fallback = True
            stage1_qasm = stage1_fallback(raw_qasm)
            fidelity = checksum1.fidelity if checksum1 else 0.0
            reason = checksum1.reason if checksum1 else "reconstruction error"
        else:
            fidelity = checksum1.fidelity
            reason = ""

        self.diagnostics.record(DiagnosticEvent(
            stage="stage1", source_file=source_name,
            passed_model=not stage1_used_fallback, fallback_used=stage1_used_fallback,
            fidelity=fidelity, reason=reason,
        ))

        # ---- Stage 2 -------------------------------------------------
        stage1_graph = qasm_to_graph(stage1_qasm)
        stage1_graph.data = stage1_graph.data.to(self.device)

        with torch.no_grad():
            predicted_keep = self.stage2.predict_keep_mask(stage1_graph.data, stage1_graph.node_qubits, stage1_graph.n_qubits).cpu()

        stage2_used_fallback = False
        checksum2_fidelity: float | None = None

        if predicted_keep.all():
            # Model predicts nothing to prune — balanced circuit == stage1 output.
            # No dimension change, so Checksum 2 (which is specifically about
            # cross-dimension partial-trace verification) doesn't apply; Checksum 1
            # already certified this circuit's fidelity against the raw input.
            balanced_qasm, kept_qubits = stage1_qasm, list(range(stage1_graph.n_qubits))
        else:
            try:
                balanced_qasm, kept_qubits = apply_stage2_mask(stage1_qasm, predicted_keep)
                checksum2 = run_checksum2(raw_qasm, balanced_qasm, kept_qubits)
            except Exception as exc:  # noqa: BLE001
                checksum2 = None
                logger.warning("Stage 2 reconstruction raised %s — routing to fallback", exc)

            if checksum2 is None or not checksum2.passed:
                stage2_used_fallback = True
                balanced_qasm, kept_qubits = stage2_fallback(stage1_qasm)
                checksum2_fidelity = checksum2.fidelity if checksum2 else 0.0
                reason = checksum2.reason if checksum2 else "reconstruction error"
            else:
                checksum2_fidelity = checksum2.fidelity
                reason = ""

            self.diagnostics.record(DiagnosticEvent(
                stage="stage2", source_file=source_name,
                passed_model=not stage2_used_fallback, fallback_used=stage2_used_fallback,
                fidelity=checksum2_fidelity or 0.0, reason=reason,
            ))

        return OptimizationResult(
            balanced_qasm=balanced_qasm,
            n_qubits_in=graph.n_qubits,
            n_qubits_out=len(kept_qubits),
            stage1_used_fallback=stage1_used_fallback,
            stage2_used_fallback=stage2_used_fallback,
            checksum1_fidelity=fidelity,
            checksum2_fidelity=checksum2_fidelity,
        )
