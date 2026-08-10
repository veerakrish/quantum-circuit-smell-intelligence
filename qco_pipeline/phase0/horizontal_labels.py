"""
Per-raw-node action labels for Stage 1 supervision.

Problem this solves: Qiskit's transpile() (horizontal_pairs.reduce_horizontal)
gives an authoritative Y_horiz QASM, but it does not preserve a mapping back to
*which raw gate became what* — so it can't directly supervise a node-classifier
over the raw circuit's graph. Reverse-engineering that alignment from opaque
transpiler internals is fragile.

Instead we derive an EXACT, 1:1-aligned label per raw-circuit node with a small,
local rewrite-rule pass that implements literally the two operations the spec
calls "horizontal reduction": inverse-gate cancellation and adjacent rotation
merging. Because it's local and rule-based, every raw node gets exactly one
label with no ambiguity, and it never has to "guess" what Qiskit did internally.

Qiskit's transpile() output remains the ground-truth QASM used for training
Stage 1's *sequence* target and for Checksum-1 fallback; this module supplies
the complementary node-level classification target used to train the GNN.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from qiskit import QuantumCircuit, qasm2

SELF_INVERSE_GATES = {"x", "y", "z", "h", "cx", "cy", "cz", "swap", "id"}
MERGEABLE_ROTATION_GATES = {"rz", "rx", "ry"}


class Action(IntEnum):
    KEEP = 0
    CANCEL = 1            # this gate and its inverse partner both vanish
    MERGE_INTO_PREV = 2   # this gate's angle is absorbed into the previous KEEP node


@dataclass
class NodeLabel:
    action: Action
    merged_angle: float = 0.0  # only meaningful for the KEEP node a MERGE target lands on


def label_horizontal_actions(raw_qasm: str) -> list[NodeLabel]:
    circ = qasm2.loads(raw_qasm)
    n = len(circ.data)
    labels = [NodeLabel(Action.KEEP) for _ in range(n)]

    last_node_on_wire: dict[int, int] = {}  # qubit -> most recent *un-cancelled* node id
    cancelled: set[int] = set()

    for node_id, instruction in enumerate(circ.data):
        op = instruction.operation
        qubits = [circ.find_bit(q).index for q in instruction.qubits]
        if op.name in ("measure", "barrier") or not qubits:
            last_node_on_wire.update({q: node_id for q in qubits})
            continue

        # A candidate "adjacent" partner exists only if every qubit this gate
        # touches was last touched by the SAME previous node (no intervening
        # op on any of those wires) — that's what makes the pair truly adjacent.
        prev_candidates = {last_node_on_wire.get(q) for q in qubits}
        prev_id = next(iter(prev_candidates)) if len(prev_candidates) == 1 else None

        if prev_id is not None and prev_id not in cancelled:
            prev_instruction = circ.data[prev_id]
            prev_op = prev_instruction.operation
            prev_qubits = [circ.find_bit(q).index for q in prev_instruction.qubits]

            same_wires = prev_qubits == qubits
            is_self_inverse_pair = (
                same_wires
                and op.name == prev_op.name
                and op.name in SELF_INVERSE_GATES
                and list(op.params) == list(prev_op.params)
            )
            is_mergeable_rotation = (
                same_wires
                and len(qubits) == 1
                and op.name == prev_op.name
                and op.name in MERGEABLE_ROTATION_GATES
            )

            if is_self_inverse_pair:
                labels[prev_id] = NodeLabel(Action.CANCEL)
                labels[node_id] = NodeLabel(Action.CANCEL)
                cancelled.update({prev_id, node_id})
                # After a cancellation, the wire's "last active node" reverts
                # to whatever preceded the cancelled pair (may enable further
                # cascading cancellation — this loop is intentionally simple
                # and single-pass; run it to a fixed point for full cascades).
                continue

            if is_mergeable_rotation:
                merged_theta = float(prev_op.params[0]) + float(op.params[0])
                labels[prev_id] = NodeLabel(Action.KEEP, merged_angle=merged_theta)
                labels[node_id] = NodeLabel(Action.MERGE_INTO_PREV)
                last_node_on_wire.update({q: prev_id for q in qubits})
                continue

        last_node_on_wire.update({q: node_id for q in qubits})

    return labels


def label_horizontal_actions_fixed_point(raw_qasm: str, max_passes: int = 8) -> list[NodeLabel]:
    """
    Cascading cancellation: H,H,H,H should fully cancel, not just the first
    pair. Repeatedly re-run the single pass on the surviving (non-CANCEL,
    non-MERGE_INTO_PREV) subsequence until no new cancellation/merge is found,
    then splice labels back onto the original node indices.
    """
    circ = qasm2.loads(raw_qasm)
    n = len(circ.data)
    final = [Action.KEEP] * n
    merged_angle = [0.0] * n
    alive_indices = list(range(n))

    for _ in range(max_passes):
        sub_circ = QuantumCircuit(circ.num_qubits, circ.num_clbits)
        for i in alive_indices:
            instr = circ.data[i]
            sub_circ.append(instr.operation, instr.qubits, instr.clbits)

        sub_labels = label_horizontal_actions(qasm2.dumps(sub_circ))
        changed = False
        new_alive: list[int] = []
        for local_i, orig_i in enumerate(alive_indices):
            lbl = sub_labels[local_i]
            if lbl.action == Action.CANCEL:
                final[orig_i] = Action.CANCEL
                changed = True
            elif lbl.action == Action.MERGE_INTO_PREV:
                final[orig_i] = Action.MERGE_INTO_PREV
                changed = True
            else:
                merged_angle[orig_i] = lbl.merged_angle
                new_alive.append(orig_i)

        alive_indices = new_alive
        if not changed:
            break

    return [NodeLabel(final[i], merged_angle[i]) for i in range(n)]
