"""
Top-level CLI for the smell-intelligence pipeline.

    python -m qcs_pipeline.cli mine        --raw-dir DIR --out mined_pairs.jsonl
    python -m qcs_pipeline.cli mine-kaggle --dataset-root DIR --out mined_pairs.jsonl [--opt-level N] [--n-workers N]
    python -m qcs_pipeline.cli build-rules --mined mined_pairs.jsonl --out rules.json [--min-frequency N]
    python -m qcs_pipeline.cli optimize    --rules rules.json --circuit path/to/circuit.qasm [--out out.qasm]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_mine(args: argparse.Namespace) -> None:
    from qcs_pipeline.mining.build_mined_dataset import build
    build(args.raw_dir, args.out)


def cmd_mine_kaggle(args: argparse.Namespace) -> None:
    from qcs_pipeline.mining.from_kaggle_pairs import mine_from_parquet
    mine_from_parquet(
        args.dataset_root, args.out,
        opt_level_filter=args.opt_level,
        max_rows_per_chunk=args.max_rows_per_chunk,
        n_workers=args.n_workers,
    )


def cmd_build_rules(args: argparse.Namespace) -> None:
    from qcs_pipeline.rules.rule_database import build_rule_database
    db = build_rule_database(args.mined, min_frequency=args.min_frequency)
    db.to_json(args.out)
    entries = db.entries()
    n_conflict = sum(e.conflict for e in entries)
    logger.info(
        "Rule database built: %d unique rules (%d flagged conflicting, min_frequency=%d) -> %s",
        len(entries), n_conflict, args.min_frequency, args.out,
    )


def cmd_optimize(args: argparse.Namespace) -> None:
    from qcs_pipeline.pipeline import QuantumCircuitSmellOptimizer

    optimizer = QuantumCircuitSmellOptimizer(args.rules)
    raw_qasm = args.circuit.read_text()
    result = optimizer.optimize(raw_qasm)

    logger.info(
        "Optimized %s: %d -> %d gates (fidelity=%.10f, %d rule application(s), %d quarantined, %d simulations)",
        args.circuit.name, result.n_gates_before, result.n_gates_after,
        result.fidelity, len(result.applied_smells), result.quarantined_rule_count, result.n_simulations,
    )

    if args.out:
        args.out.write_text(result.optimized_qasm)
        logger.info("Wrote optimized circuit to %s", args.out)
    else:
        print(result.optimized_qasm)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Quantum Circuit Smell Intelligence")
    sub = parser.add_subparsers(dest="command", required=True)

    p_mine = sub.add_parser("mine", help="Step 1: mine raw/transpiled diff patterns")
    p_mine.add_argument("--raw-dir", type=Path, required=True)
    p_mine.add_argument("--out", type=Path, required=True)
    p_mine.set_defaults(func=cmd_mine)

    p_mine_kaggle = sub.add_parser("mine-kaggle", help="Step 1 (fast path): mine from pre-transpiled Kaggle parquet pairs")
    p_mine_kaggle.add_argument("--dataset-root", type=Path, required=True)
    p_mine_kaggle.add_argument("--out", type=Path, required=True)
    p_mine_kaggle.add_argument("--opt-level", type=int, default=None)
    p_mine_kaggle.add_argument("--max-rows-per-chunk", type=int, default=None)
    p_mine_kaggle.add_argument("--n-workers", type=int, default=1)
    p_mine_kaggle.set_defaults(func=cmd_mine_kaggle)

    p_rules = sub.add_parser("build-rules", help="Step 2: canonicalize mined patterns into a rule database")
    p_rules.add_argument("--mined", type=Path, required=True)
    p_rules.add_argument("--out", type=Path, required=True)
    p_rules.add_argument("--min-frequency", type=int, default=2)
    p_rules.set_defaults(func=cmd_build_rules)

    p_opt = sub.add_parser("optimize", help="Apply the smell detector + verified repair to one circuit")
    p_opt.add_argument("--rules", type=Path, required=True)
    p_opt.add_argument("--circuit", type=Path, required=True)
    p_opt.add_argument("--out", type=Path, default=None)
    p_opt.set_defaults(func=cmd_optimize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
