"""
Checksum 1 — Stage 1 gate: same-Hilbert-space state fidelity.

Raw and Stage-1-output circuits share the same qubit count, so a direct
state-vector inner product is valid (no partial trace needed here — that's
Checksum 2's problem, once qubit count changes).

Fidelity = |<psi_raw | psi_stage1>|^2  >=  1 - 1e-6

Assumption (documented, not hidden): both circuits are evaluated from the
all-zero input state |0...0>, which is the standard convention for circuit-
optimization benchmarks (including MNISQ). If your circuits encode data via
non-trivial input states, swap Statevector(circ) for Statevector(circ,
initial_state=your_state) in both branches below.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector, state_fidelity

logger = logging.getLogger(__name__)

FIDELITY_THRESHOLD = 1 - 1e-6
MAX_QUBITS_FOR_STATEVECTOR = 24  # 2^24 complex amplitudes ~ 256MB; raise/lower per hardware


@dataclass
class Checksum1Result:
    passed: bool
    fidelity: float
    n_qubits: int
    reason: str = ""


def _strip_measurements(circ: QuantumCircuit) -> QuantumCircuit:
    stripped = circ.copy()
    stripped.remove_final_measurements(inplace=True)
    return stripped


def run_checksum1(raw_qasm: str, stage1_qasm: str) -> Checksum1Result:
    raw_circ = _strip_measurements(qasm2.loads(raw_qasm))
    stage1_circ = _strip_measurements(qasm2.loads(stage1_qasm))

    if raw_circ.num_qubits != stage1_circ.num_qubits:
        return Checksum1Result(
            passed=False,
            fidelity=0.0,
            n_qubits=raw_circ.num_qubits,
            reason=(
                f"Qubit count changed in Stage 1 ({raw_circ.num_qubits} -> "
                f"{stage1_circ.num_qubits}); Stage 1 must never prune wires. "
                "Reject and route to classical fallback."
            ),
        )

    if raw_circ.num_qubits > MAX_QUBITS_FOR_STATEVECTOR:
        return Checksum1Result(
            passed=False,
            fidelity=0.0,
            n_qubits=raw_circ.num_qubits,
            reason=(
                f"{raw_circ.num_qubits} qubits exceeds statevector verification budget "
                f"({MAX_QUBITS_FOR_STATEVECTOR}); use a sampling-based fidelity estimator "
                "or classically-verifiable subcircuit decomposition instead."
            ),
        )

    try:
        psi_raw = Statevector.from_instruction(raw_circ)
        psi_stage1 = Statevector.from_instruction(stage1_circ)
    except Exception as exc:  # noqa: BLE001
        return Checksum1Result(passed=False, fidelity=0.0, n_qubits=raw_circ.num_qubits, reason=f"Simulation error: {exc}")

    fidelity = state_fidelity(psi_raw, psi_stage1)
    passed = fidelity >= FIDELITY_THRESHOLD

    if not passed:
        logger.warning(
            "Checksum 1 FAILED: fidelity=%.10f < threshold=%.10f (n_qubits=%d)",
            fidelity, FIDELITY_THRESHOLD, raw_circ.num_qubits,
        )

    return Checksum1Result(
        passed=passed,
        fidelity=fidelity,
        n_qubits=raw_circ.num_qubits,
        reason="" if passed else f"fidelity {fidelity:.10f} below threshold {FIDELITY_THRESHOLD:.10f}",
    )
