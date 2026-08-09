"""
Wire-adjacency-aware pattern matching — the correctness-critical component.

Unlike the Step 1 miner (which diffs raw circuit.data LIST order, an
approximation flagged in mining/pair_diff.py), matching here is done against
TRUE wire adjacency: gate B is only considered "next after" gate A on qubit q
if B is literally the next operation touching q, regardless of how many
gates on OTHER, non-interacting qubits sit between them in the circuit's
instruction list. This is what makes it safe to apply a mined rule to a
circuit the miner never saw — matches are found by genuine dependency
structure, not by incidental serialization order.
"""
from __future__ import annotations

from dataclasses import dataclass

from qcs_pipeline.mining.gate_sequence import GateOp
from qcs_pipeline.rules.canonicalize import CanonicalOp


@dataclass
class WireDAG:
    ops: list[GateOp]
    next_on_wire: dict[tuple[int, int], int | None]  # (node_id, qubit) -> next node_id touching qubit, or None
    first_on_wire: dict[int, int | None]              # qubit -> first node_id touching it, or None


def build_wire_dag(ops: list[GateOp]) -> WireDAG:
    last_on_wire: dict[int, int] = {}
    first_on_wire: dict[int, int | None] = {}
    next_on_wire: dict[tuple[int, int], int | None] = {}

    for node_id, op in enumerate(ops):
        for q in op.qubits:
            if q not in first_on_wire:
                first_on_wire[q] = node_id
            if q in last_on_wire:
                next_on_wire[(last_on_wire[q], q)] = node_id
            last_on_wire[q] = node_id

    for q, node_id in last_on_wire.items():
        next_on_wire.setdefault((node_id, q), None)

    return WireDAG(ops=ops, next_on_wire=next_on_wire, first_on_wire=first_on_wire)


def _assign_roles(role_map: dict[int, int], pattern_qubits: tuple[int, ...], actual_qubits: tuple[int, ...]) -> bool:
    if len(pattern_qubits) != len(actual_qubits):
        return False
    for role, actual_qubit in zip(pattern_qubits, actual_qubits):
        if role in role_map:
            if role_map[role] != actual_qubit:
                return False
        else:
            if actual_qubit in role_map.values():
                return False  # injective: two distinct roles can't collapse onto the same physical qubit
            role_map[role] = actual_qubit
    return True


@dataclass
class Match:
    node_ids: list[int]        # matched circuit node ids, in pattern order
    role_map: dict[int, int]   # pattern relative-qubit-role -> actual circuit qubit index


def find_matches(dag: WireDAG, pattern: tuple[CanonicalOp, ...]) -> list[Match]:
    """All non-overlapping-candidate matches of `pattern` in `dag`. Overlap
    resolution (picking a consistent, non-conflicting subset) is the caller's
    job — see smell_detector.apply_rules."""
    matches: list[Match] = []
    if not pattern:
        return matches

    for start_id, op in enumerate(dag.ops):
        if op.name != pattern[0].name or len(op.qubits) != len(pattern[0].qubits):
            continue

        role_map: dict[int, int] = {}
        if not _assign_roles(role_map, pattern[0].qubits, op.qubits):
            continue

        matched_ids = [start_id]
        last_on_wire = {q: start_id for q in op.qubits}
        ok = True

        for pat_op in pattern[1:]:
            expected_qubits = [role_map[r] for r in pat_op.qubits if r in role_map]
            new_role_count = sum(1 for r in pat_op.qubits if r not in role_map)

            if not expected_qubits or new_role_count > 0:
                # Pattern op introduces a qubit role with no established wire
                # anchor yet — our mined local patterns are always fully
                # connected to the first gate's qubits, so this indicates a
                # pattern we can't safely resolve via wire adjacency; skip it.
                ok = False
                break

            candidate_id = None
            consistent = True
            for aq in expected_qubits:
                nxt = dag.next_on_wire.get((last_on_wire[aq], aq))
                if nxt is None:
                    consistent = False
                    break
                if candidate_id is None:
                    candidate_id = nxt
                elif candidate_id != nxt:
                    consistent = False  # the expected qubits' next-gates disagree — not a single joint gate
                    break

            if not consistent or candidate_id is None:
                ok = False
                break

            candidate_op = dag.ops[candidate_id]
            if candidate_op.name != pat_op.name or len(candidate_op.qubits) != len(pat_op.qubits):
                ok = False
                break
            if not _assign_roles(role_map, pat_op.qubits, candidate_op.qubits):
                ok = False
                break

            matched_ids.append(candidate_id)
            for q in candidate_op.qubits:
                last_on_wire[q] = candidate_id

        if ok:
            matches.append(Match(node_ids=matched_ids, role_map=dict(role_map)))

    return matches
