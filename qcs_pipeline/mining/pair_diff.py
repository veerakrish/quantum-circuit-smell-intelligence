"""
Step 1 — mine (raw, transpiled) pairs into removal/rewrite pattern records.

Approach: diff the raw circuit's GateOp sequence against the transpiled one
using difflib.SequenceMatcher (the same LCS-based alignment algorithm behind
`diff`/`git diff`), then extract every non-'equal' opcode block as a mined
pattern, with a few 'equal' gates of context on either side.

KNOWN LIMITATION (documented, not hidden): this diffs the circuits' raw
instruction-list ORDER, not true per-wire adjacency. Qiskit's transpiler
mostly preserves relative gate order, so in practice mined windows are
correct for the common case, but a transpiler pass that reorders
non-interacting gates across wires could produce a spurious mined pattern.
This is why the DETECTOR (detector/wire_matcher.py) re-verifies true wire
adjacency independently before ever applying a mined rule to a new circuit —
the miner only needs to be a good enough source of CANDIDATE patterns; it is
not the layer responsible for correctness.

SECOND KNOWN LIMITATION, specific to opt_level>=2 targets: Qiskit's
ConsolidateBlocks + UnitarySynthesis passes resynthesize an entire 2-qubit
interaction block into an unrelated gate sequence. difflib sees that as one
big non-'equal' opcode covering the whole block, so naively it gets mined as
a single "removed -> replacement" pattern spanning dozens of gates. That is
NOT a generalizable local identity — it only holds for that one example's
specific rotation angles, not for the same gate names appearing anywhere
else. `MAX_PATTERN_LEN` below rejects any mined window (removed or
replacement side) longer than this, which is what actually filters those
whole-block resynthesis artifacts out at the source, independent of which
opt_level the pairs came from — a real local rewrite (gate cancellation,
adjacent rotation merge, small commutation) is a handful of gates; a
resynthesized block is not.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from qcs_pipeline import qasm_compat
from qcs_pipeline.mining.gate_sequence import GateOp, circuit_to_gate_ops

CONTEXT_WINDOW = 2  # gates of 'equal' context kept on each side of a mined pattern
MAX_PATTERN_LEN = 8  # reject removed/replacement windows longer than this (see module docstring)


@dataclass
class MinedPattern:
    context_before: list[GateOp]
    removed: list[GateOp]     # present in raw, absent (or replaced) in transpiled
    replacement: list[GateOp]  # present in transpiled at this position (empty for pure deletion)
    context_after: list[GateOp]
    source_file: str = ""


@dataclass
class MiningReport:
    source_file: str
    n_gates_raw: int
    n_gates_transpiled: int
    n_gates_removed: int
    removed_gate_names: list[str]
    patterns: list[MinedPattern] = field(default_factory=list)
    n_oversized_skipped: int = 0  # non-'equal' opcodes rejected for exceeding MAX_PATTERN_LEN


def mine_pair(
    raw_qasm: str, transpiled_qasm: str, source_file: str = "", max_pattern_len: int = MAX_PATTERN_LEN,
) -> MiningReport:
    raw_circ = qasm_compat.loads(raw_qasm)
    transpiled_circ = qasm_compat.loads(transpiled_qasm)

    raw_ops = circuit_to_gate_ops(raw_circ)
    transpiled_ops = circuit_to_gate_ops(transpiled_circ)

    raw_keys = [op.diff_key() for op in raw_ops]
    transpiled_keys = [op.diff_key() for op in transpiled_ops]

    matcher = difflib.SequenceMatcher(a=raw_keys, b=transpiled_keys, autojunk=False)
    opcodes = matcher.get_opcodes()

    patterns: list[MinedPattern] = []
    removed_gate_names: list[str] = []
    n_oversized_skipped = 0

    for idx, (tag, a0, a1, b0, b1) in enumerate(opcodes):
        if tag == "equal":
            continue

        removed = raw_ops[a0:a1]
        replacement = transpiled_ops[b0:b1]

        if len(removed) > max_pattern_len or len(replacement) > max_pattern_len:
            # Almost certainly a whole resynthesized block (ConsolidateBlocks +
            # UnitarySynthesis), not a generalizable local identity — see the
            # module docstring's second known limitation. Skip mining it rather
            # than feed the rule database a rule that can only ever match the
            # exact circuit it was mined from.
            n_oversized_skipped += 1
            continue

        removed_gate_names.extend(op.name for op in removed)

        context_before = _tail_context(opcodes, idx, raw_ops, transpiled_ops, side="before")
        context_after = _head_context(opcodes, idx, raw_ops, transpiled_ops, side="after")

        patterns.append(MinedPattern(
            context_before=context_before,
            removed=removed,
            replacement=replacement,
            context_after=context_after,
            source_file=source_file,
        ))

    return MiningReport(
        source_file=source_file,
        n_gates_raw=len(raw_ops),
        n_gates_transpiled=len(transpiled_ops),
        n_gates_removed=len(raw_ops) - len(transpiled_ops),
        removed_gate_names=removed_gate_names,
        patterns=patterns,
        n_oversized_skipped=n_oversized_skipped,
    )


def _tail_context(opcodes, idx, raw_ops, transpiled_ops, side: str) -> list[GateOp]:
    if idx == 0:
        return []
    prev_tag, pa0, pa1, pb0, pb1 = opcodes[idx - 1]
    if prev_tag != "equal":
        return []  # only take context from genuinely shared ('equal') gates
    src = raw_ops[pa0:pa1]
    return src[-CONTEXT_WINDOW:]


def _head_context(opcodes, idx, raw_ops, transpiled_ops, side: str) -> list[GateOp]:
    if idx + 1 >= len(opcodes):
        return []
    next_tag, na0, na1, nb0, nb1 = opcodes[idx + 1]
    if next_tag != "equal":
        return []
    src = raw_ops[na0:na1]
    return src[:CONTEXT_WINDOW]
