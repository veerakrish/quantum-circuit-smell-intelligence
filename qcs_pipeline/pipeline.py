"""
End-to-end smell-intelligence pipeline:

    raw QASM
        │
        ▼
   Smell detector (wire-adjacency-aware rule matching against RuleDatabase)
        │  candidate matches
        ▼
   Bisection repair (exact fidelity check; quarantine unsound rule applications)
        │
        ▼
   optimized QASM (gate cancellation/merges applied, exact-fidelity verified)

Steps 1 & 2 (mining + rule database construction) are offline, run once via
`qcs_pipeline.mining.build_mined_dataset` and `rules.rule_database.build_rule_database`
— see cli.py. This module is the online/inference side: apply an already-built
rule database to a (possibly never-before-seen) circuit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from qiskit import QuantumCircuit

from qcs_pipeline.detector.smell_detector import Smell, detect, find_candidate_matches
from qcs_pipeline.rules.rule_database import RuleDatabase
from qcs_pipeline.verification.bisect_repair import RepairResult, verify_and_repair

logger = logging.getLogger(__name__)


@dataclass
class SmellOptimizationResult:
    optimized_qasm: str
    n_gates_before: int
    n_gates_after: int
    fidelity: float
    applied_smells: list[Smell]
    quarantined_rule_count: int
    n_simulations: int


class QuantumCircuitSmellOptimizer:
    def __init__(self, rule_db_path: Path):
        self.db = RuleDatabase.from_json(rule_db_path)
        logger.info("Loaded rule database: %d unique rules (%d flagged as conflicting)",
                    len(self.db.entries()), sum(e.conflict for e in self.db.entries()))

    def detect_smells(self, raw_qasm: str) -> list[Smell]:
        """Review-only: flag matches without touching the circuit."""
        circ = QuantumCircuit.from_qasm_str(raw_qasm)
        smells, _ = detect(circ, self.db)
        return smells

    def optimize(self, raw_qasm: str) -> SmellOptimizationResult:
        raw_circ = QuantumCircuit.from_qasm_str(raw_qasm)
        n_before = sum(raw_circ.count_ops().values())

        candidates = find_candidate_matches(raw_circ, self.db)
        repair: RepairResult = verify_and_repair(raw_circ, candidates)

        n_after = sum(repair.final_circuit.count_ops().values())
        if repair.quarantined:
            logger.warning("%d rule application(s) quarantined this run — see result.quarantined_rule_count", len(repair.quarantined))

        return SmellOptimizationResult(
            optimized_qasm=repair.final_circuit.qasm(),
            n_gates_before=n_before,
            n_gates_after=n_after,
            fidelity=repair.fidelity,
            applied_smells=repair.accepted_smells,
            quarantined_rule_count=len(repair.quarantined),
            n_simulations=repair.n_simulations,
        )
