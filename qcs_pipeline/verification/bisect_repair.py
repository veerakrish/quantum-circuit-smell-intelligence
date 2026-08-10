"""
Exact-fidelity verification + bisection repair.

Reuses qco_pipeline's Checksum 1 (same-qubit-count state fidelity) — gate
cancellation/merging never changes qubit count, so that check applies
directly here without modification.

Key principle (stated explicitly because it's what makes bisection valid,
not just convenient): a TRUE gate identity, applied to a disjoint set of
circuit positions, composes safely with any other true identity applied
elsewhere — they don't interact. So if applying the FULL set of matched
rewrites breaks fidelity, the failure is attributable to one or more
individually-unsound matches, not to some emergent interaction between
otherwise-valid ones. That's what licenses divide-and-conquer: split the
match set in half, repair each half independently against the ORIGINAL raw
circuit, and simply union the two halves' safe results back together.

This is deliberately NOT a numerical "calibrate until close enough" loop —
a sound rule must reproduce fidelity 1.0 (within floating point), not merely
"close." A failure means a specific rule is wrong (mined from a context-
dependent transpiler artifact, or a canonicalization bug) and should be
quarantined, not nudged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from qiskit import QuantumCircuit, qasm2

from qco_pipeline.verification.checksum1 import Checksum1Result, run_checksum1
from qcs_pipeline.detector.smell_detector import Smell, matches_to_smells, rewrite_with_matches
from qcs_pipeline.detector.wire_matcher import Match
from qcs_pipeline.rules.rule_database import RuleEntry

logger = logging.getLogger(__name__)


@dataclass
class RepairResult:
    final_circuit: QuantumCircuit
    accepted_smells: list[Smell]
    quarantined: list[tuple[RuleEntry, str]]  # (rule, reason) for every match found unsound
    n_simulations: int
    fidelity: float


def verify_and_repair(raw_circ: QuantumCircuit, candidate_matches: list[tuple[Match, RuleEntry]]) -> RepairResult:
    stats = {"n_simulations": 0}
    accepted, quarantined = _bisect(raw_circ, candidate_matches, stats)

    final_circuit = rewrite_with_matches(raw_circ, accepted)
    final_check = run_checksum1(qasm2.dumps(raw_circ), qasm2.dumps(final_circuit))
    stats["n_simulations"] += 1

    if not final_check.passed:
        # Should not happen if _bisect's invariant holds (every accepted
        # match individually verified safe, and identities compose) — but
        # verify the FINAL assembled circuit too, defensively, rather than
        # trusting composition blindly forever.
        logger.error(
            "Post-bisection recomposition still failed fidelity (%.10f) — falling back to the "
            "raw circuit unmodified. This indicates two mined rules interact in a way the "
            "compositional-safety assumption doesn't cover; both should be reviewed.",
            final_check.fidelity,
        )
        return RepairResult(
            final_circuit=raw_circ, accepted_smells=[], quarantined=quarantined,
            n_simulations=stats["n_simulations"], fidelity=final_check.fidelity,
        )

    return RepairResult(
        final_circuit=final_circuit,
        accepted_smells=matches_to_smells(accepted),
        quarantined=quarantined,
        n_simulations=stats["n_simulations"],
        fidelity=final_check.fidelity,
    )


def _check(raw_circ: QuantumCircuit, matches: list[tuple[Match, RuleEntry]], stats: dict) -> Checksum1Result:
    candidate_circ = rewrite_with_matches(raw_circ, matches)
    stats["n_simulations"] += 1
    return run_checksum1(qasm2.dumps(raw_circ), qasm2.dumps(candidate_circ))


def _bisect(
    raw_circ: QuantumCircuit,
    matches: list[tuple[Match, RuleEntry]],
    stats: dict,
) -> tuple[list[tuple[Match, RuleEntry]], list[tuple[RuleEntry, str]]]:
    if not matches:
        return [], []

    result = _check(raw_circ, matches, stats)
    if result.passed:
        return matches, []

    if len(matches) == 1:
        match, entry = matches[0]
        reason = f"isolated application fails fidelity ({result.fidelity:.10f} < threshold): {result.reason}"
        logger.warning("Quarantining rule %s: %s", [op.name for op in entry.pattern], reason)
        return [], [(entry, reason)]

    mid = len(matches) // 2
    left, right = matches[:mid], matches[mid:]

    left_accepted, left_quarantined = _bisect(raw_circ, left, stats)
    right_accepted, right_quarantined = _bisect(raw_circ, right, stats)

    return left_accepted + right_accepted, left_quarantined + right_quarantined
