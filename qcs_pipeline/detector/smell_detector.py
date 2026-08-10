"""
Apply a RuleDatabase to a circuit: find matches via true wire adjacency,
resolve overlaps, evaluate symbolic param relations against the ACTUAL
matched gates' params, and rewrite.

Analogous to a code-smell linter's "detect findings, then auto-fix" flow —
`detect()` alone gives you flagged smells without touching the circuit
(useful for review/audit), `apply()` produces the rewritten circuit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from qiskit import QuantumCircuit

from qcs_pipeline.detector.wire_matcher import Match, WireDAG, build_wire_dag, find_matches
from qcs_pipeline.mining.gate_sequence import GateOp, circuit_to_gate_ops, gate_ops_to_circuit
from qcs_pipeline.rules.rule_database import RuleDatabase, RuleEntry

_PARAM_TOKEN_RE = re.compile(r"p(\d+)_(\d+)")


@dataclass
class Smell:
    rule_pattern_names: list[str]
    matched_node_ids: list[int]
    matched_qubits: list[int]
    frequency: int  # how many times this rule was observed during mining — a confidence proxy


def detect(circ: QuantumCircuit, db: RuleDatabase) -> tuple[list[Smell], WireDAG]:
    ops = circuit_to_gate_ops(circ)
    dag = build_wire_dag(ops)
    index = db.lookup_by_first_gate()

    smells: list[Smell] = []
    for entry in db.entries():
        matches = find_matches(dag, entry.pattern)
        for m in matches:
            smells.append(Smell(
                rule_pattern_names=[op.name for op in entry.pattern],
                matched_node_ids=m.node_ids,
                matched_qubits=sorted(set(m.role_map.values())),
                frequency=entry.frequency,
            ))
    return smells, dag


def _resolve_overlaps(candidates: list[tuple[Match, RuleEntry]]) -> list[tuple[Match, RuleEntry]]:
    """Greedy selection: prefer longer patterns, then higher-frequency (more
    trusted) rules, then earliest position; skip anything overlapping an
    already-accepted match's node set."""
    ranked = sorted(
        candidates,
        key=lambda pair: (-len(pair[0].node_ids), -pair[1].frequency, min(pair[0].node_ids)),
    )
    used_nodes: set[int] = set()
    accepted: list[tuple[Match, RuleEntry]] = []
    for match, entry in ranked:
        if used_nodes.intersection(match.node_ids):
            continue
        accepted.append((match, entry))
        used_nodes.update(match.node_ids)
    return accepted


_TERM_RE = re.compile(r"([+-]?)p(\d+)_(\d+)")


def _eval_param_relation(expr: str, removed_ops: list[GateOp]) -> float:
    """
    Evaluates the tiny fixed grammar canonicalize.py ever produces:
    an optional leading sign, then one or more `p{i}_{j}` terms joined by
    +/- (e.g. "p0_0", "-p0_0", "p0_0+p1_0", "p1_0-p0_0"). Deliberately no
    eval()/exec() — rule files should be treated as data, not code, even
    though today's canonicalize.py only ever emits this exact shape.
    """
    if expr == "UNRESOLVED":
        raise ValueError("Attempted to apply a rule with an unresolved param relation — this rule is structural-only.")
    if expr == "0":
        return 0.0

    terms = _TERM_RE.findall(expr)
    if not terms or sum(len(t[0]) + len(t[1]) + len(t[2]) + 1 for t in terms) < len(expr.replace(" ", "")):
        raise ValueError(f"Unrecognized param relation expression: {expr!r}")

    total = 0.0
    for sign, i, j in terms:
        value = removed_ops[int(i)].params[int(j)]
        total += -value if sign == "-" else value
    return total


def is_structural_only(entry: RuleEntry) -> bool:
    """True if any rewrite-side gate has an UNRESOLVED param relation — i.e.
    canonicalize.py could not express its angle as a simple copy/negate/sum
    of the removed gates' angles (common for rz->rz rewrites, where Qiskit
    resynthesized a whole gate sequence into a new closed-form angle rather
    than a linear combination of the old ones). Such rules are real,
    frequently-observed patterns, but we cannot safely compute the replacement
    parameter — applying them would require guessing a value, which
    _eval_param_relation refuses to do. Kept in the database for detect()/audit
    visibility; never selected as an applicable candidate."""
    return any("UNRESOLVED" in op.param_relation for op in entry.rewrite)


def find_candidate_matches(circ: QuantumCircuit, db: RuleDatabase) -> list[tuple[Match, RuleEntry]]:
    """All accepted (non-overlapping, non-conflicting, fully-resolved) rule
    matches for a circuit — the unit of work `verification/bisect_repair.py`
    bisects over."""
    ops = circuit_to_gate_ops(circ)
    dag = build_wire_dag(ops)

    candidates: list[tuple[Match, RuleEntry]] = []
    for entry in db.entries():
        if entry.conflict or is_structural_only(entry):
            continue  # ambiguous or param-unresolved rules are surfaced via detect(), never auto-applied
        for m in find_matches(dag, entry.pattern):
            candidates.append((m, entry))

    return _resolve_overlaps(candidates)


def summarize_applicability(db: RuleDatabase) -> dict[str, int]:
    """
    Breaks down a RuleDatabase's entries by whether they can actually be
    auto-applied. A raw `len(db.entries())` overstates what's usable — many
    real, frequently-observed rewrites (rz->rz being the biggest offender)
    have angle changes canonicalize.py couldn't express symbolically, and
    those are never selected by find_candidate_matches regardless of how
    often they were mined.
    """
    entries = db.entries()
    n_conflict = sum(1 for e in entries if e.conflict)
    n_structural_only = sum(1 for e in entries if not e.conflict and is_structural_only(e))
    n_applicable = len(entries) - n_conflict - n_structural_only
    return {
        "total": len(entries),
        "conflicting": n_conflict,
        "structural_only": n_structural_only,
        "applicable": n_applicable,
    }


def rewrite_with_matches(circ: QuantumCircuit, accepted: list[tuple[Match, RuleEntry]]) -> QuantumCircuit:
    """
    Apply exactly the given (already-overlap-resolved) matches to `circ`.
    Public and reused by bisect_repair.py with arbitrary SUBSETS of a full
    match list — that's precisely what makes bisection possible: re-running
    this with half the matches to isolate which one broke fidelity.
    """
    ops = circuit_to_gate_ops(circ)
    accepted_node_ids = {node_id for match, _ in accepted for node_id in match.node_ids}

    # Rebuild: walk original ops in order; when we hit the FIRST node of an
    # accepted match, splice in its rewrite; skip all other nodes belonging
    # to that match (matches not in `accepted` are left untouched).
    match_at_first_node = {min(m.node_ids): (m, entry) for m, entry in accepted}
    skip_nodes = accepted_node_ids - set(match_at_first_node.keys())

    new_ops: list[GateOp] = []
    for node_id, op in enumerate(ops):
        if node_id in skip_nodes:
            continue
        if node_id in match_at_first_node:
            match, entry = match_at_first_node[node_id]
            removed_actual = [ops[i] for i in match.node_ids]
            for rewrite_op in entry.rewrite:
                actual_qubits = tuple(match.role_map[r] for r in rewrite_op.qubits)
                params = tuple(_eval_param_relation(expr, removed_actual) for expr in rewrite_op.param_relation)
                new_ops.append(GateOp(name=rewrite_op.name, qubits=actual_qubits, params=params))
            continue
        new_ops.append(op)

    return gate_ops_to_circuit(new_ops, n_qubits=circ.num_qubits, n_clbits=circ.num_clbits)


def matches_to_smells(accepted: list[tuple[Match, RuleEntry]]) -> list[Smell]:
    return [
        Smell(
            rule_pattern_names=[o.name for o in entry.pattern],
            matched_node_ids=match.node_ids,
            matched_qubits=sorted(set(match.role_map.values())),
            frequency=entry.frequency,
        )
        for match, entry in accepted
    ]


def apply_rules(circ: QuantumCircuit, db: RuleDatabase) -> tuple[QuantumCircuit, list[Smell]]:
    accepted = find_candidate_matches(circ, db)
    new_circ = rewrite_with_matches(circ, accepted)
    return new_circ, matches_to_smells(accepted)
