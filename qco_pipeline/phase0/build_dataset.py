"""
Phase 0 orchestrator: raw MNISQ QASM directory -> Dataset 1 (horizontal_pairs.jsonl)
and Dataset 2 (vertical_pairs.jsonl) on disk, ready for Stage 1 / Stage 2 training.

Run:
    python -m qco_pipeline.phase0.build_dataset \
        --raw-dir data/mnisq_raw \
        --out-dir data/phase0
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from qco_pipeline.phase0.horizontal_pairs import reduce_horizontal
from qco_pipeline.phase0.vertical_pairs import prune_idle_wires

logger = logging.getLogger(__name__)


def build(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    horiz_path = out_dir / "horizontal_pairs.jsonl"
    vert_path = out_dir / "vertical_pairs.jsonl"

    qasm_files = sorted(raw_dir.glob("*.qasm"))
    if not qasm_files:
        raise FileNotFoundError(f"No .qasm files found under {raw_dir}")

    n_ok, n_failed = 0, 0
    with horiz_path.open("w") as hf, vert_path.open("w") as vf:
        for qasm_file in tqdm(qasm_files, desc="Phase 0"):
            try:
                raw_qasm = qasm_file.read_text()

                h_pair = reduce_horizontal(raw_qasm)
                hf.write(json.dumps({**asdict(h_pair), "source_file": qasm_file.name}) + "\n")

                v_pair = prune_idle_wires(h_pair.horiz_qasm)
                # raw_qasm carried through explicitly so Checksum 2 (which needs
                # the ORIGINAL n-qubit circuit for its partial trace, not just
                # the n-qubit horizontal intermediate) never has to guess it.
                vf.write(json.dumps({**asdict(v_pair), "raw_qasm": raw_qasm, "source_file": qasm_file.name}) + "\n")

                n_ok += 1
            except Exception as exc:  # noqa: BLE001 — Phase 0 must never crash the whole batch
                n_failed += 1
                logger.warning("Skipping %s: %s", qasm_file.name, exc)

    logger.info("Phase 0 complete: %d ok, %d failed, written to %s", n_ok, n_failed, out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build Phase 0 horizontal/vertical training pairs")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
