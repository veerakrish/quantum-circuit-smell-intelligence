"""Turn Stage 2's per-wire keep mask into a re-indexed, executable QASM circuit."""
from __future__ import annotations

import torch
from qiskit import QuantumCircuit, qasm2


def apply_stage2_mask(horiz_qasm: str, keep_mask: torch.Tensor) -> tuple[str, list[int]]:
    """
    Returns (balanced_qasm, kept_qubits) where kept_qubits are the ORIGINAL
    indices retained, listed in their new contiguous order — this mapping is
    exactly what Checksum 2's permutation-alignment step needs.
    """
    circ = qasm2.loads(horiz_qasm)
    n = circ.num_qubits

    keep_mask_list = keep_mask.tolist()
    kept_qubits = [q for q in range(n) if keep_mask_list[q]]
    if not kept_qubits:
        raise ValueError("Stage 2 pruned every wire — refusing to emit a 0-qubit circuit; reject and fall back.")

    remap = {old: new for new, old in enumerate(kept_qubits)}
    balanced = QuantumCircuit(len(kept_qubits), circ.num_clbits)

    for instruction in circ.data:
        op = instruction.operation
        old_qargs = [circ.find_bit(q).index for q in instruction.qubits]
        if any(q not in remap for q in old_qargs):
            # Model predicted "prune" for a wire that a gate still touches —
            # this is exactly the case Checksum 2 must catch. We do not
            # silently drop the gate; we surface it as a hard error so the
            # sample is routed to the classical fallback instead of emitting
            # a semantically wrong circuit.
            raise ValueError(
                f"Stage 2 predicted prune for qubit(s) {[q for q in old_qargs if q not in remap]} "
                f"still touched by gate {op.name} — inconsistent prediction, must fall back."
            )
        new_qargs = [remap[q] for q in old_qargs]
        balanced.append(op, new_qargs, instruction.clbits)

    return qasm2.dumps(balanced), kept_qubits
