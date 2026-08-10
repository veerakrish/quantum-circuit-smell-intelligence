"""
Turn Stage 1's per-node predictions back into an executable QASM circuit.

This is the model-output -> circuit reconstruction step that Checksum 1
(verification/checksum1.py) consumes.
"""
from __future__ import annotations

import torch
from qiskit import QuantumCircuit, qasm2

from qco_pipeline.phase0.horizontal_labels import Action


def apply_stage1_actions(raw_qasm: str, action_logits: torch.Tensor, predicted_angle: torch.Tensor) -> str:
    circ = qasm2.loads(raw_qasm)
    actions = action_logits.argmax(dim=-1).tolist()

    out = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    for node_id, instruction in enumerate(circ.data):
        action = Action(actions[node_id])
        if action in (Action.CANCEL, Action.MERGE_INTO_PREV):
            continue  # gate removed / absorbed into an earlier KEEP node

        op = instruction.operation
        if action == Action.KEEP and op.name in ("rz", "rx", "ry") and float(predicted_angle[node_id]) != 0.0:
            # If this KEEP node was the target of a merge, its angle was
            # predicted separately from the raw op's own angle — regression
            # target is the *merged* rotation angle, not a delta.
            merged = float(predicted_angle[node_id])
            if abs(merged) > 1e-9:
                op = op.copy()
                op.params[0] = merged

        out.append(op, instruction.qubits, instruction.clbits)

    return qasm2.dumps(out)
