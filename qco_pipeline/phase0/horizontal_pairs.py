"""
Dataset 1 (Horizontal Pairs) generation.

X_raw  -> raw MNISQ OpenQASM circuit
Y_horiz -> horizontally reduced circuit: inverse-gate cancellation (H*H=I, CX*CX=I),
           merged rotation angles (RZ(a)*RZ(b) -> RZ(a+b)), commutation-based
           depth reduction. Qubit COUNT is preserved (n == n) — no wire pruning here,
           that is Stage 2's job (see vertical_pairs.py).

Ground truth is produced with a deterministic *classical* compiler, never with the
model under training — this is what keeps Stage 1's supervision signal exact.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import standard_gates  # noqa: F401 (documents gate set assumption)

logger = logging.getLogger(__name__)


@dataclass
class HorizontalPair:
    raw_qasm: str
    horiz_qasm: str
    n_qubits: int
    raw_gate_count: int
    horiz_gate_count: int
    raw_depth: int
    horiz_depth: int


def reduce_horizontal(raw_qasm: str, optimization_level: int = 3, basis_gates: list[str] | None = None) -> HorizontalPair:
    """
    Deterministically produce the horizontally-reduced target for one raw circuit.

    Qiskit's transpiler at optimization_level=3 runs the passes we need for the
    horizontal target without touching qubit count:
      - Optimize1qGatesDecomposition / Optimize1qGates    (single-qubit merges)
      - CommutativeCancellation                            (H*H, CX*CX, etc.)
      - CXCancellation / InverseCancellation
      - ConsolidateBlocks + UnitarySynthesis (only within a wire's own block,
        never removes a wire)

    `basis_gates=None` keeps Qiskit's default universal basis so gate identities
    (e.g. RZ merging) are not obscured by an unrelated re-decomposition.
    """
    raw_circ = QuantumCircuit.from_qasm_str(raw_qasm)
    n_qubits = raw_circ.num_qubits

    horiz_circ = transpile(
        raw_circ,
        optimization_level=optimization_level,
        basis_gates=basis_gates,
        seed_transpiler=0,  # determinism — same input always yields same target
    )

    if horiz_circ.num_qubits != n_qubits:
        # Should never happen at this stage — transpile() alone doesn't prune wires.
        raise RuntimeError(
            f"Horizontal reduction changed qubit count ({n_qubits} -> "
            f"{horiz_circ.num_qubits}); this belongs in Stage 2, not Stage 1. "
            "Check that no ancilla-removal pass leaked into this transpile call."
        )

    return HorizontalPair(
        raw_qasm=raw_qasm,
        horiz_qasm=horiz_circ.qasm(),
        n_qubits=n_qubits,
        raw_gate_count=sum(raw_circ.count_ops().values()),
        horiz_gate_count=sum(horiz_circ.count_ops().values()),
        raw_depth=raw_circ.depth(),
        horiz_depth=horiz_circ.depth(),
    )


def reduce_horizontal_pyzx(raw_qasm: str) -> str:
    """
    Optional alternate/cross-check target generator using PyZX's ZX-calculus
    simplification (full_reduce). Useful for building a *second* teacher signal
    to sanity-check Qiskit's target, or as the fallback compiler referenced in
    Checksum 1 (see fallback/classical_fallback.py).
    """
    import pyzx as zx

    g = zx.Circuit.from_qasm(raw_qasm).to_graph()
    zx.full_reduce(g, quiet=True)
    g.normalize()
    out_circ = zx.extract_circuit(g.copy())
    return out_circ.to_qasm()
