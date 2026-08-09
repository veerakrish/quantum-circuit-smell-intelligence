"""
Dataset 2 (Vertical Pairs) generation.

Y_horiz    -> horizontally optimized circuit (n qubits, from horizontal_pairs.py)
Y_balanced -> idle / zero-contribution wires pruned, remaining k < n active wires
              re-indexed to a contiguous register [0, 1, ..., k-1].

"Idle" here means a qubit whose removal provably does not change the reduced
density matrix on the remaining wires — the touch-graph heuristic below is the
*deterministic, sound* subset of that condition (an untouched wire is always
separable and traceable-out at zero cost); it does not attempt the harder,
generally-intractable case of detecting an *entangled-but-still-separable-after-
gates-cancel* wire. That harder case is left to Checksum 2 to catch/reject if a
future heuristic over-claims a prune.
"""
from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit


@dataclass
class VerticalPair:
    horiz_qasm: str
    balanced_qasm: str
    n_qubits_in: int
    n_qubits_out: int
    kept_qubits: list[int]  # original indices retained, in their new contiguous order


def _touched_qubits(circ: QuantumCircuit) -> set[int]:
    touched: set[int] = set()
    for instruction in circ.data:
        for qubit in instruction.qubits:
            touched.add(circ.find_bit(qubit).index)
    return touched


def prune_idle_wires(horiz_qasm: str) -> VerticalPair:
    """
    Deterministic wire-pruning target generator.

    1. Identify qubits with zero gate operations (never appear in any Instruction).
    2. Remove those wires from the register.
    3. Re-index the remaining wires to [0 .. k-1], preserving relative order.
    """
    circ = QuantumCircuit.from_qasm_str(horiz_qasm)
    n = circ.num_qubits

    touched = _touched_qubits(circ)
    kept_qubits = sorted(touched)  # relative order preserved
    k = len(kept_qubits)

    if k == n:
        # Nothing to prune — balanced target equals the horizontal input.
        return VerticalPair(
            horiz_qasm=horiz_qasm,
            balanced_qasm=horiz_qasm,
            n_qubits_in=n,
            n_qubits_out=n,
            kept_qubits=list(range(n)),
        )

    remap = {old: new for new, old in enumerate(kept_qubits)}
    balanced = QuantumCircuit(k, circ.num_clbits)

    for instruction in circ.data:
        op = instruction.operation
        old_qargs = [circ.find_bit(q).index for q in instruction.qubits]
        new_qargs = [remap[q] for q in old_qargs]
        balanced.append(op, new_qargs, instruction.clbits)

    return VerticalPair(
        horiz_qasm=horiz_qasm,
        balanced_qasm=balanced.qasm(),
        n_qubits_in=n,
        n_qubits_out=k,
        kept_qubits=kept_qubits,
    )
