"""
Aggregate canonical rules mined across the whole dataset into a rule table:
dedup, frequency-rank, flag conflicts, and persist to/from JSON.

A "conflict" is the same canonical PATTERN mapping to more than one distinct
canonical REWRITE across different mined examples. This is not necessarily a
bug — e.g. an RZ-merge pattern's param_relation is always the structural
"p0_0+p1_0" regardless of literal values, so true RZ merges should NOT
conflict; a conflict more often means the context-free pattern (2+ gates,
no surrounding context) is ambiguous and actually needs more context to
resolve, or the miner picked up a spurious/context-dependent transpiler
artifact. Conflicting rules are kept but ranked by frequency, and the
low-frequency variants are flagged for manual review rather than silently
dropped or silently trusted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from qcs_pipeline.rules.canonicalize import CanonicalOp, CanonicalRule, pattern_key, rewrite_key


@dataclass
class RuleEntry:
    pattern: tuple[CanonicalOp, ...]
    rewrite: tuple[CanonicalOp, ...]
    frequency: int
    conflict: bool = False  # True if this pattern also maps to other rewrites elsewhere in the table
    example_sources: list[str] = field(default_factory=list)


class RuleDatabase:
    def __init__(self):
        # pattern_key -> { rewrite_key -> RuleEntry }
        self._table: dict[tuple, dict[tuple, RuleEntry]] = {}

    def add(self, rule: CanonicalRule) -> None:
        pkey = pattern_key(rule)
        rkey = rewrite_key(rule)
        bucket = self._table.setdefault(pkey, {})
        if rkey in bucket:
            bucket[rkey].frequency += 1
            if rule.example_source:
                bucket[rkey].example_sources.append(rule.example_source)
        else:
            bucket[rkey] = RuleEntry(
                pattern=rule.pattern, rewrite=rule.rewrite, frequency=1,
                example_sources=[rule.example_source] if rule.example_source else [],
            )
        # Recompute conflict flags for this pattern bucket.
        is_conflict = len(bucket) > 1
        for entry in bucket.values():
            entry.conflict = is_conflict

    def finalize(self, min_frequency: int = 1) -> None:
        """Drop rules seen fewer than `min_frequency` times — cheap noise filter."""
        for pkey in list(self._table.keys()):
            bucket = self._table[pkey]
            for rkey in list(bucket.keys()):
                if bucket[rkey].frequency < min_frequency:
                    del bucket[rkey]
            if not bucket:
                del self._table[pkey]

    def entries(self) -> list[RuleEntry]:
        return [entry for bucket in self._table.values() for entry in bucket.values()]

    def lookup_by_first_gate(self) -> dict[str, list[RuleEntry]]:
        """Group rules by their pattern's first gate name — the index the
        detector uses to avoid testing every rule at every position."""
        index: dict[str, list[RuleEntry]] = {}
        for entry in self.entries():
            first_gate = entry.pattern[0].name
            index.setdefault(first_gate, []).append(entry)
        # Longer patterns first within each bucket — greedy longest-match wins,
        # same convention as most peephole optimizers / regex engines.
        for bucket in index.values():
            bucket.sort(key=lambda e: (-len(e.pattern), -e.frequency))
        return index

    def to_json(self, path: Path) -> None:
        payload = []
        for entry in self.entries():
            payload.append({
                "pattern": [_op_to_dict(op) for op in entry.pattern],
                "rewrite": [_op_to_dict(op) for op in entry.rewrite],
                "frequency": entry.frequency,
                "conflict": entry.conflict,
                "example_sources": entry.example_sources[:10],  # cap for file size
            })
        payload.sort(key=lambda r: -r["frequency"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "RuleDatabase":
        db = cls()
        rows = json.loads(path.read_text())
        for row in rows:
            pattern = tuple(_dict_to_op(d) for d in row["pattern"])
            rewrite = tuple(_dict_to_op(d) for d in row["rewrite"])
            entry = RuleEntry(
                pattern=pattern, rewrite=rewrite, frequency=row["frequency"],
                conflict=row["conflict"], example_sources=row.get("example_sources", []),
            )
            db._table.setdefault(pattern, {})[rewrite] = entry
        return db


def rule_id(entry: RuleEntry) -> str:
    """Stable, human-readable identity for a rule entry: pattern gate
    sequence (with relative qubit roles) plus rewrite gate sequence. Used to
    aggregate per-rule accept/quarantine telemetry across many circuits in
    Stage 3 without relying on Python object identity (which breaks once a
    RuleDatabase is reloaded from JSON, e.g. across notebook cells)."""
    pattern_sig = "|".join(f"{op.name}{op.qubits}" for op in entry.pattern)
    rewrite_sig = "|".join(f"{op.name}{op.qubits}" for op in entry.rewrite)
    return f"{pattern_sig} -> {rewrite_sig}"


def _op_to_dict(op: CanonicalOp) -> dict:
    return {"name": op.name, "qubits": list(op.qubits), "param_relation": list(op.param_relation)}


def _dict_to_op(d: dict) -> CanonicalOp:
    return CanonicalOp(name=d["name"], qubits=tuple(d["qubits"]), param_relation=tuple(d["param_relation"]))


def build_rule_database(mined_dataset_path: Path, min_frequency: int = 2) -> RuleDatabase:
    """Read Step 1's mined_pairs.jsonl, canonicalize every pattern, and build
    the deduped, frequency-ranked rule table (Step 2)."""
    import json as _json

    from qcs_pipeline.mining.gate_sequence import GateOp
    from qcs_pipeline.rules.canonicalize import canonicalize_pattern

    db = RuleDatabase()
    with mined_dataset_path.open() as f:
        for line in f:
            row = _json.loads(line)
            for pattern in row["mined_patterns"]:
                removed = [GateOp(o["name"], tuple(o["qubits"]), tuple(o["params"])) for o in pattern["removed"]]
                replacement = [GateOp(o["name"], tuple(o["qubits"]), tuple(o["params"])) for o in pattern["replacement"]]
                rule = canonicalize_pattern(removed, replacement, source=row["source_file"])
                if rule is not None:
                    db.add(rule)

    db.finalize(min_frequency=min_frequency)
    return db
