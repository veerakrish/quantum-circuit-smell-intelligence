"""
Adapter for the veerukhannan/mnisq-optbench-pairs Kaggle dataset — its
`pairs_k*.parquet` chunk files already contain (input_qasm, target_qasm)
pairs produced by Qiskit's transpiler (columns include `opt_level` and
`transpiler_seed`, matching phase0/horizontal_pairs.reduce_horizontal's own
transpile() call). This means Step 1's expensive part — actually running the
transpiler — is already done; this module just re-diffs the already-paired
QASM strings into mined patterns, skipping build_mined_dataset.py's
"read raw .qasm file -> transpile it yourself" path entirely.

Chunks are processed one at a time and released before the next is loaded,
so peak memory stays bounded by a single ~150 MB parquet chunk, not the
dataset's full 17+ GB.
"""
from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from qcs_pipeline.mining.build_mined_dataset import _pattern_to_dict
from qcs_pipeline.mining.pair_diff import mine_pair

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["base_id", "opt_level", "transpiler_seed", "input_qasm", "target_qasm"]


def find_pair_chunks(dataset_root: Path) -> list[Path]:
    """Locate pairs_k*.parquet files regardless of exact nesting under
    dataset_root — the Kaggle mount path nests them under final/pairs/, but
    globbing recursively avoids hardcoding that and breaking on a dataset
    version bump."""
    matches = sorted(Path(p) for p in glob.glob(str(dataset_root / "**" / "pairs_k*.parquet"), recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"No pairs_k*.parquet files found under {dataset_root} — "
            "confirm the Kaggle dataset is attached and check the mounted "
            "folder name with `glob.glob('/kaggle/input/*')`."
        )
    return matches


def mine_from_parquet(
    dataset_root: Path,
    out_path: Path,
    opt_level_filter: int | None = None,
    max_rows_per_chunk: int | None = None,
) -> None:
    """
    opt_level_filter: keep only rows with this opt_level (recommended — mixing
        optimization levels in one rule database conflates different transpiler
        behaviors under the same canonical patterns). Inspect the actual
        distribution first (`pd.read_parquet(chunk).opt_level.value_counts()`)
        before picking a value.
    max_rows_per_chunk: cap rows read per chunk file — useful for a fast first
        pass within Kaggle's session time limit before committing to a full run.
    """
    chunk_files = find_pair_chunks(dataset_root)
    logger.info("Found %d parquet chunk file(s) under %s", len(chunk_files), dataset_root)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_failed = 0, 0

    with out_path.open("w") as f:
        for chunk_path in tqdm(chunk_files, desc="Mining from Kaggle parquet chunks"):
            df = pd.read_parquet(chunk_path, columns=REQUIRED_COLUMNS)

            if opt_level_filter is not None:
                df = df[df["opt_level"] == opt_level_filter]
            if max_rows_per_chunk is not None:
                df = df.head(max_rows_per_chunk)

            for row in df.itertuples(index=False):
                source_id = f"{chunk_path.name}:{row.base_id}"
                try:
                    report = mine_pair(row.input_qasm, row.target_qasm, source_file=source_id)
                    f.write(json.dumps({
                        "source_file": source_id,
                        "raw_qasm": row.input_qasm,
                        "transpiled_qasm": row.target_qasm,
                        "gates_removed_count": report.n_gates_removed,
                        "gates_removed_list": report.removed_gate_names,
                        "mined_patterns": [_pattern_to_dict(p) for p in report.patterns],
                    }) + "\n")
                    n_ok += 1
                except Exception as exc:  # noqa: BLE001
                    n_failed += 1
                    logger.debug("Skipping %s: %s", source_id, exc)

            del df  # release this chunk's memory before the next iteration loads one

    logger.info("Kaggle-parquet mining complete: %d ok, %d failed -> %s", n_ok, n_failed, out_path)
