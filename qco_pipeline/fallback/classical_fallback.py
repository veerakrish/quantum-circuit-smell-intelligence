"""
Deterministic classical fallback, triggered whenever a checksum fails.

Rule 2 from the skill spec: "Zero Tolerance for State Hallucination — never
feed Stage 1 outputs into Stage 2 without passing Checksum 1 verification."
This module is what a failed checksum routes to instead of the (untrusted)
model output, plus structured diagnostics logging so failures are triage-able
rather than silently swallowed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from qco_pipeline.phase0.horizontal_pairs import reduce_horizontal
from qco_pipeline.phase0.vertical_pairs import prune_idle_wires

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticEvent:
    stage: str                 # "stage1" | "stage2"
    source_file: str
    passed_model: bool
    fallback_used: bool
    fidelity: float
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DiagnosticsLog:
    """Append-only JSONL diagnostics sink — one line per pipeline decision."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: DiagnosticEvent) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event.__dict__) + "\n")
        if event.fallback_used:
            logger.warning(
                "[%s] FALLBACK TRIGGERED for %s (fidelity=%.10f): %s",
                event.stage, event.source_file, event.fidelity, event.reason,
            )

    def summarize(self) -> dict:
        if not self.path.exists():
            return {"total": 0}
        stages: dict[str, dict[str, int]] = {}
        with self.path.open() as f:
            for line in f:
                row = json.loads(line)
                s = stages.setdefault(row["stage"], {"total": 0, "model_passed": 0, "fallback_used": 0})
                s["total"] += 1
                s["model_passed"] += int(row["passed_model"])
                s["fallback_used"] += int(row["fallback_used"])
        return stages


def stage1_fallback(raw_qasm: str) -> str:
    """Classical horizontal reduction — same code path used to build Dataset 1's
    ground truth (horizontal_pairs.reduce_horizontal), so it is fidelity-safe
    by construction and needs no re-verification against Checksum 1."""
    return reduce_horizontal(raw_qasm).horiz_qasm


def stage2_fallback(horiz_qasm: str) -> tuple[str, list[int]]:
    """Classical idle-wire pruning — same code path used to build Dataset 2's
    ground truth (vertical_pairs.prune_idle_wires)."""
    v_pair = prune_idle_wires(horiz_qasm)
    return v_pair.balanced_qasm, v_pair.kept_qubits
