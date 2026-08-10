"""
Adapter for the veerukhannan/mnisq-optbench-pairs Kaggle dataset — its
`pairs_k*.parquet` chunk files already contain (input_qasm, target_qasm)
pairs produced by Qiskit's transpiler (columns include `opt_level` and
`transpiler_seed`, matching phase0/horizontal_pairs.reduce_horizontal's own
transpile() call). This means Step 1's expensive part — actually running the
transpiler — is already done; this module just re-diffs the already-paired
QASM strings into mined patterns, skipping build_mined_dataset.py's
"read raw .qasm file -> transpile it yourself" path entirely.

Chunks are processed one at a time per worker and released before the next
is loaded, so peak memory stays bounded by roughly `n_workers` chunks
(~150 MB each) in memory at once, not the dataset's full 17+ GB.

Mining does no quantum simulation at all (that only happens later, during
verification/optimize()) — it's pure QASM parsing + difflib diffing, which
is why this parallelizes cleanly across processes: every row's diff is
independent of every other row, with no shared state to synchronize.

Output rows deliberately omit the full raw_qasm/transpiled_qasm text —
build_rule_database() only ever reads `mined_patterns`, so storing complete
circuit QASM per row (some MNISQ-OptBench circuits run into the thousands
of gates) is pure duplication of what's already in the source parquet, and
at Kaggle scale (up to ~60k rows) is enough to exhaust the output disk
quota on its own. `source_file` (chunk name + base_id) is kept so any row
can still be looked up in the original parquet if needed.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _mine_chunk_group(
    chunk_files: list[Path],
    out_path: Path,
    opt_level_filter: int | None,
    max_rows_per_chunk: int | None,
    show_progress: bool,
) -> tuple[int, int, int, int]:
    """
    Mine a subset of chunk files into `out_path`. This is the unit of work
    each worker process (or the single sequential run) executes — module-level
    so it's picklable/importable by ProcessPoolExecutor workers, unlike a
    closure defined inline in a notebook cell.

    Returns (n_ok, n_failed, n_patterns_mined, n_oversized_skipped) — the last
    two make the pair_diff.MAX_PATTERN_LEN filter's effect visible in the log
    instead of silently shrinking the rule database.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_failed, n_patterns_mined, n_oversized_skipped = 0, 0, 0, 0

    iterator = tqdm(chunk_files, desc=f"Mining ({out_path.name})") if show_progress else chunk_files

    with out_path.open("w") as f:
        for chunk_path in iterator:
            df = pd.read_parquet(chunk_path, columns=REQUIRED_COLUMNS)

            if opt_level_filter is not None:
                df = df[df["opt_level"] == opt_level_filter]
            if max_rows_per_chunk is not None:
                df = df.head(max_rows_per_chunk)

            for row in df.itertuples(index=False):
                source_id = f"{chunk_path.name}:{row.base_id}"
                try:
                    report = mine_pair(row.input_qasm, row.target_qasm, source_file=source_id)
                    # Deliberately NOT storing raw_qasm/transpiled_qasm here —
                    # build_rule_database() only ever reads `mined_patterns`,
                    # so the full circuit text (up to thousands of gates each)
                    # would be pure duplication of what's already sitting in
                    # the source parquet, multiplied by every mined row. That
                    # redundancy is what filled /kaggle/working's disk quota.
                    # `source_file` (chunk name + base_id) is enough to look
                    # a specific row back up in the parquet if ever needed.
                    f.write(json.dumps({
                        "source_file": source_id,
                        "gates_removed_count": report.n_gates_removed,
                        "gates_removed_list": report.removed_gate_names,
                        "mined_patterns": [_pattern_to_dict(p) for p in report.patterns],
                    }) + "\n")
                    n_ok += 1
                    n_patterns_mined += len(report.patterns)
                    n_oversized_skipped += report.n_oversized_skipped
                except Exception as exc:  # noqa: BLE001
                    n_failed += 1
                    logger.debug("Skipping %s: %s", source_id, exc)

            del df  # release this chunk's memory before the next iteration loads one

    return n_ok, n_failed, n_patterns_mined, n_oversized_skipped


def _split_round_robin(items: list, n_groups: int) -> list[list]:
    """Round-robin rather than contiguous slicing, so each worker gets a mix
    of chunk indices instead of e.g. worker 0 always getting the first N —
    guards against any ordering-correlated size skew across chunk files."""
    groups: list[list] = [[] for _ in range(n_groups)]
    for i, item in enumerate(items):
        groups[i % n_groups].append(item)
    return groups


def mine_from_parquet(
    dataset_root: Path,
    out_path: Path,
    opt_level_filter: int | None = None,
    max_rows_per_chunk: int | None = None,
    n_workers: int = 1,
) -> None:
    """
    opt_level_filter: keep only rows with this opt_level (recommended — mixing
        optimization levels in one rule database conflates different transpiler
        behaviors under the same canonical patterns). Inspect the actual
        distribution first (`pd.read_parquet(chunk).opt_level.value_counts()`)
        before picking a value.
    max_rows_per_chunk: cap rows read per chunk file — useful for a fast first
        pass within Kaggle's session time limit before committing to a full run.
    n_workers: number of worker processes. Mining does no simulation, so this
        is pure CPU-count-bound parallelism — check `os.cpu_count()` and don't
        request more workers than that, or you'll oversubscribe and add
        context-switching overhead instead of speeding anything up. n_workers=1
        (default) runs sequentially with no multiprocessing overhead at all.
    """
    chunk_files = find_pair_chunks(dataset_root)
    logger.info("Found %d parquet chunk file(s) under %s", len(chunk_files), dataset_root)

    available = os.cpu_count() or 1
    if n_workers > available:
        logger.warning(
            "n_workers=%d exceeds os.cpu_count()=%d — this will oversubscribe "
            "and likely run SLOWER than n_workers=%d due to context switching. "
            "Capping to %d.",
            n_workers, available, available, available,
        )
        n_workers = available

    if n_workers <= 1:
        n_ok, n_failed, n_patterns, n_oversized = _mine_chunk_group(
            chunk_files, out_path, opt_level_filter, max_rows_per_chunk, show_progress=True,
        )
        logger.info(
            "Kaggle-parquet mining complete: %d ok, %d failed, %d patterns mined, "
            "%d oversized windows skipped (> pair_diff.MAX_PATTERN_LEN) -> %s",
            n_ok, n_failed, n_patterns, n_oversized, out_path,
        )
        return

    groups = _split_round_robin(chunk_files, n_workers)
    part_paths = [out_path.with_suffix(f".part{i}.jsonl") for i in range(n_workers)]

    logger.info("Mining with %d worker processes across %d chunk files", n_workers, len(chunk_files))
    results: list[tuple[int, int, int, int]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_mine_chunk_group, group, part_path, opt_level_filter, max_rows_per_chunk, i == 0): i
            for i, (group, part_path) in enumerate(zip(groups, part_paths))
            if group  # skip empty groups (more workers than chunk files)
        }
        for future in as_completed(futures):
            worker_id = futures[future]
            n_ok, n_failed, n_patterns, n_oversized = future.result()  # re-raises if the worker hit an unhandled exception
            results.append((n_ok, n_failed, n_patterns, n_oversized))
            logger.info(
                "Worker %d done: %d ok, %d failed, %d patterns mined, %d oversized skipped",
                worker_id, n_ok, n_failed, n_patterns, n_oversized,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as merged:
        for part_path in part_paths:
            if not part_path.exists():
                continue
            with part_path.open() as part:
                merged.writelines(part)
            part_path.unlink()

    total_ok = sum(r[0] for r in results)
    total_failed = sum(r[1] for r in results)
    total_patterns = sum(r[2] for r in results)
    total_oversized = sum(r[3] for r in results)
    logger.info(
        "Kaggle-parquet mining complete (parallel, %d workers): %d ok, %d failed, %d patterns mined, "
        "%d oversized windows skipped (> pair_diff.MAX_PATTERN_LEN) -> %s",
        n_workers, total_ok, total_failed, total_patterns, total_oversized, out_path,
    )
