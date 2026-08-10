"""
Step 1 CLI: MNISQ raw QASM directory -> mined_pairs.jsonl

Each output row matches the schema you specified:
    raw_qasm, transpiled_qasm, gates_removed_count, gates_removed_list,
    per_qubit_pattern (aligned before/after gate-name sequence per qubit)
plus the full mined pattern windows (context + removed + replacement) that
Step 2 canonicalizes into rules.

Run:
    python -m qcs_pipeline.mining.build_mined_dataset \
        --raw-dir data/mnisq_raw --out data/mined_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from qiskit import qasm2
from tqdm import tqdm

from qco_pipeline.phase0.horizontal_pairs import reduce_horizontal
from qcs_pipeline.mining.gate_sequence import circuit_to_gate_ops
from qcs_pipeline.mining.pair_diff import MinedPattern, mine_pair

logger = logging.getLogger(__name__)


def _gate_op_to_dict(op) -> dict:
    return {"name": op.name, "qubits": list(op.qubits), "params": list(op.params)}


def _pattern_to_dict(p: MinedPattern) -> dict:
    return {
        "context_before": [_gate_op_to_dict(o) for o in p.context_before],
        "removed": [_gate_op_to_dict(o) for o in p.removed],
        "replacement": [_gate_op_to_dict(o) for o in p.replacement],
        "context_after": [_gate_op_to_dict(o) for o in p.context_after],
    }


def _per_qubit_pattern(raw_circ: QuantumCircuit, transpiled_circ: QuantumCircuit) -> dict[int, dict]:
    """For each qubit, the aligned before/after gate-NAME sequence — a quick
    human-readable summary distinct from the full mined pattern windows."""
    n = raw_circ.num_qubits
    result = {}
    for q in range(n):
        before = [i.operation.name for i in raw_circ.data if q in [raw_circ.find_bit(x).index for x in i.qubits]]
        after = [i.operation.name for i in transpiled_circ.data if q in [transpiled_circ.find_bit(x).index for x in i.qubits]]
        result[q] = {"before": before, "after": after}
    return result


def build(raw_dir: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qasm_files = sorted(raw_dir.glob("*.qasm"))
    if not qasm_files:
        raise FileNotFoundError(f"No .qasm files under {raw_dir}")

    n_ok, n_failed = 0, 0
    with out_path.open("w") as f:
        for qasm_file in tqdm(qasm_files, desc="Mining raw/transpiled pairs"):
            try:
                raw_qasm = qasm_file.read_text()
                h_pair = reduce_horizontal(raw_qasm)  # deterministic Qiskit transpile — same code path as qco_pipeline
                transpiled_qasm = h_pair.horiz_qasm

                raw_circ = qasm2.loads(raw_qasm)
                transpiled_circ = qasm2.loads(transpiled_qasm)

                report = mine_pair(raw_qasm, transpiled_qasm, source_file=qasm_file.name)

                row = {
                    "source_file": qasm_file.name,
                    "raw_qasm": raw_qasm,
                    "transpiled_qasm": transpiled_qasm,
                    "gates_removed_count": report.n_gates_removed,
                    "gates_removed_list": report.removed_gate_names,
                    "per_qubit_pattern": _per_qubit_pattern(raw_circ, transpiled_circ),
                    "mined_patterns": [_pattern_to_dict(p) for p in report.patterns],
                }
                f.write(json.dumps(row) + "\n")
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                logger.warning("Skipping %s: %s", qasm_file.name, exc)

    logger.info("Step 1 mining complete: %d ok, %d failed -> %s", n_ok, n_failed, out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Step 1: mine raw/transpiled circuit pairs")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.raw_dir, args.out)


if __name__ == "__main__":
    main()
