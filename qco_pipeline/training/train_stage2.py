"""
Stage 2 training loop — mirrors train_stage1.py's structure, but validates
against Checksum 2 (partial-trace fidelity across the dimension change)
instead of Checksum 1.

Run:
    python -m qco_pipeline.training.train_stage2 \
        --dataset data/phase0/vertical_pairs.jsonl \
        --out checkpoints/stage2.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from qco_pipeline.graph.circuit_graph import qasm_to_graph
from qco_pipeline.models.apply_stage2 import apply_stage2_mask
from qco_pipeline.models.stage2_vertical import Stage2VerticalModel
from qco_pipeline.training.losses import Stage2BalancedLoss, compute_stage2_pos_weight
from qco_pipeline.verification.checksum2 import run_checksum2

logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["n_qubits_out"] < row["n_qubits_in"]:  # only train on rows that actually prune something
                rows.append(row)
    return rows


def build_example(row: dict):
    graph = qasm_to_graph(row["horiz_qasm"])
    n = row["n_qubits_in"]
    kept = set(row["kept_qubits"])
    keep_mask = torch.tensor([q in kept for q in range(n)], dtype=torch.bool)
    return row["horiz_qasm"], graph, keep_mask


def evaluate_checksum2_pass_rate(model: Stage2VerticalModel, eval_rows: list[dict], device: str, max_samples: int = 64) -> float:
    model.eval()
    sample = eval_rows[:max_samples]
    n_passed, n_total = 0, 0
    with torch.no_grad():
        for row in sample:
            horiz_qasm, graph, _ = build_example(row)
            graph.data = graph.data.to(device)
            predicted_keep = model.predict_keep_mask(graph.data, graph.node_qubits, graph.n_qubits).cpu()
            try:
                balanced_qasm, kept_qubits = apply_stage2_mask(horiz_qasm, predicted_keep)
                result = run_checksum2(row.get("raw_qasm", horiz_qasm), balanced_qasm, kept_qubits)
                n_passed += int(result.passed)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Eval sample failed to reconstruct/verify: %s", exc)
            n_total += 1
    return n_passed / max(n_total, 1)


def train(dataset_path: Path, out_path: Path, epochs: int = 30, lr: float = 3e-4, val_fraction: float = 0.1, seed: int = 0) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)

    rows = load_dataset(dataset_path)
    if not rows:
        raise ValueError(
            "No pruning-positive examples in the vertical-pairs dataset — "
            "Stage 2 has nothing to learn. Check Phase 0's idle-wire detection, "
            "or your MNISQ source circuits may simply not contain idle qubits."
        )
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    all_masks = [build_example(r)[2] for r in train_rows]
    pos_weight = compute_stage2_pos_weight(all_masks)
    logger.info("Stage 2 prune-class pos_weight: %.3f", pos_weight.item())

    model = Stage2VerticalModel().to(device)
    criterion = Stage2BalancedLoss(pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_pass_rate = -1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        random.shuffle(train_rows)

        for row in tqdm(train_rows, desc=f"Epoch {epoch + 1}/{epochs}"):
            _, graph, keep_mask = build_example(row)
            graph.data = graph.data.to(device)
            keep_mask = keep_mask.to(device)

            optimizer.zero_grad()
            prune_logit = model(graph.data, graph.node_qubits, graph.n_qubits)
            losses = criterion(prune_logit, keep_mask)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += losses["total"].item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_rows), 1)

        pass_rate = evaluate_checksum2_pass_rate(model, val_rows, device)
        logger.info("Epoch %d: train_loss=%.4f  checksum2_pass_rate=%.3f", epoch + 1, avg_loss, pass_rate)

        if pass_rate >= best_pass_rate:
            best_pass_rate = pass_rate
            torch.save({"model_state": model.state_dict(), "pos_weight": pos_weight, "pass_rate": pass_rate}, out_path)
            logger.info("  -> new best checkpoint saved (pass_rate=%.3f)", pass_rate)

    logger.info("Training complete. Best Checksum-2 pass rate: %.3f. Checkpoint: %s", best_pass_rate, out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train(args.dataset, args.out, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
