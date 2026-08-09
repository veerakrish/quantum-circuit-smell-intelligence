"""
Step 2 — canonicalize a mined pattern into a reusable rule.

Two abstractions turn a one-off diff observation into a generalizable rule:

  1. RELATIVE QUBIT ROLES — remap absolute qubit indices to 0,1,2,... by
     first appearance, so "H(q3),H(q3)" and "H(q7),H(q7)" become the same
     canonical pattern "H(0),H(0)".

  2. SYMBOLIC PARAM RELATIONS — for continuous-parameter gates (rz/rx/ry),
     don't canonicalize the literal angle away; instead detect whether the
     replacement's angle is a simple function of the removed gates' angles
     (copy / sum / difference / negate) and store that RELATION as the rule,
     not any one example's numbers. If no simple relation holds, the rule is
     kept for its structural (which-gates-to-remove) value but flagged as
     "param_relation=None" — such a rule must never be used to synthesize a
     new parameter value, only to recognize where a rewrite CAN occur.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

from qcs_pipeline.mining.gate_sequence import GateOp

PARAM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CanonicalOp:
    """A gate in canonical (relative-qubit, symbolic-param) form."""
    name: str
    qubits: tuple[int, ...]          # relative roles, e.g. (0,) or (0, 1)
    param_relation: tuple[str, ...] = ()  # one symbolic expression per param, e.g. ("p0+p1",)


@dataclass
class CanonicalRule:
    pattern: tuple[CanonicalOp, ...]     # what to search for
    rewrite: tuple[CanonicalOp, ...]     # what to replace it with
    n_qubits_spanned: int
    example_source: str = ""


def _remap_qubits(ops: list[GateOp]) -> tuple[list[GateOp], dict[int, int]]:
    remap: dict[int, int] = {}
    remapped: list[GateOp] = []
    for op in ops:
        new_qubits = []
        for q in op.qubits:
            if q not in remap:
                remap[q] = len(remap)
            new_qubits.append(remap[q])
        remapped.append(GateOp(name=op.name, qubits=tuple(new_qubits), params=op.params))
    return remapped, remap


def _infer_param_expr(target_value: float, removed_params: list[tuple[str, float]]) -> str | None:
    """
    removed_params: [(symbolic_name, concrete_value), ...] e.g. [("p0", 0.3), ("p1", 0.1)]
    Returns a symbolic expression string if `target_value` matches a simple
    combination of the removed gates' params within tolerance, else None.
    """
    if abs(target_value) < PARAM_TOLERANCE:
        return "0"

    for name, value in removed_params:
        if math.isclose(target_value, value, abs_tol=PARAM_TOLERANCE):
            return name
        if math.isclose(target_value, -value, abs_tol=PARAM_TOLERANCE):
            return f"-{name}"

    for (n1, v1), (n2, v2) in combinations(removed_params, 2):
        if math.isclose(target_value, v1 + v2, abs_tol=PARAM_TOLERANCE):
            return f"{n1}+{n2}"
        if math.isclose(target_value, v1 - v2, abs_tol=PARAM_TOLERANCE):
            return f"{n1}-{n2}"
        if math.isclose(target_value, v2 - v1, abs_tol=PARAM_TOLERANCE):
            return f"{n2}-{n1}"

    return None  # no simple relation found — rule kept structurally, param unresolved


def canonicalize_pattern(removed: list[GateOp], replacement: list[GateOp], source: str = "") -> CanonicalRule | None:
    """
    Canonicalizes the (removed -> replacement) pair. Qubit canonicalization
    is computed over BOTH sequences jointly (a rewrite can't introduce a
    qubit role that wasn't in the pattern — if it does, something is wrong
    with the mined pair and we reject it rather than guess).
    """
    if not removed:
        return None  # nothing to search for — not a usable rule

    combined, remap = _remap_qubits(removed + replacement)
    canon_removed = combined[: len(removed)]
    canon_replacement = combined[len(removed):]

    # Flat, named list of removed params for relation inference: p0, p1, ...
    removed_param_names: list[tuple[str, float]] = []
    for i, op in enumerate(removed):
        for j, val in enumerate(op.params):
            removed_param_names.append((f"p{i}_{j}", val))

    pattern_ops = tuple(
        CanonicalOp(name=op.name, qubits=op.qubits, param_relation=())  # pattern side: params irrelevant to matching structure
        for op in canon_removed
    )

    rewrite_ops = []
    for op in canon_replacement:
        relations = []
        for val in op.params:
            expr = _infer_param_expr(val, removed_param_names)
            relations.append(expr if expr is not None else "UNRESOLVED")
        rewrite_ops.append(CanonicalOp(name=op.name, qubits=op.qubits, param_relation=tuple(relations)))

    return CanonicalRule(
        pattern=pattern_ops,
        rewrite=tuple(rewrite_ops),
        n_qubits_spanned=len(remap),
        example_source=source,
    )


def pattern_key(rule: CanonicalRule) -> tuple:
    """Hashable dedup/lookup key — structure only (gate names + relative qubits)."""
    return tuple((op.name, op.qubits) for op in rule.pattern)


def rewrite_key(rule: CanonicalRule) -> tuple:
    return tuple((op.name, op.qubits, op.param_relation) for op in rule.rewrite)
