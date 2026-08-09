"""
Stage 1 training loop.

Checkpoint selection is done on CHECKSUM 1 PASS RATE on the held-out split,
not on validation loss alone — this is the practical mechanism that keeps
training "balanced" between over-aggressive rewriting (breaks fidelity) and
under-aggressive rewriting (low compression, but the loss would look fine).

Run:
    python -m qco_pipeline.training.train_stage1 \
        --dataset data/phase0/horizontal_pairs.jsonl \
        --out checkpoints/stage1.pt
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
from qco_pipeline.models.apply_stage1 import apply_stage1_actions
from qco_pipeline.models.stage1_horizontal import Stage1HorizontalModel
from qco_pipeline.phase0.horizontal_labels import label_horizontal_actions_fixed_point
from qco_pipeline.training.losses import Stage1BalancedLoss, compute_stage1_class_weights
from qco_pipeline.verification.checksum1 import run_checksum1

logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_example(row: dict):
    graph = qasm_to_graph(row["raw_qasm"])
    labels = label_horizontal_actions_fixed_point(row["raw_qasm"])

    action_targets = torch.tensor([int(lbl.action) for lbl in labels], dtype=torch.long)
    angle_targets = torch.tensor([lbl.merged_angle for lbl in labels], dtype=torch.float)
    angle_mask = torch.tensor([lbl.merged_angle != 0.0 for lbl in labels], dtype=torch.bool)

    return row["raw_qasm"], graph, action_targets, angle_targets, angle_mask


def evaluate_checksum_pass_rate(model: Stage1HorizontalModel, eval_rows: list[dict], device: str, max_samples: int = 64) -> float:
    model.eval()
    sample = eval_rows[:max_samples]
    n_passed = 0
    with torch.no_grad():
        for row in sample:
            raw_qasm, graph, *_ = build_example(row)
            graph.data = graph.data.to(device)
            action_logits, predicted_angle = model(graph.data)
            try:
                stage1_qasm = apply_stage1_actions(raw_qasm, action_logits.cpu(), predicted_angle.cpu())
                result = run_checksum1(raw_qasm, stage1_qasm)
                n_passed += int(result.passed)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Eval sample failed to reconstruct/verify: %s", exc)
    return n_passed / max(len(sample), 1)


def train(dataset_path: Path, out_path: Path, epochs: int = 30, lr: float = 3e-4, val_fraction: float = 0.1, seed: int = 0) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)

    rows = load_dataset(dataset_path)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    # Class weights computed once over the full training label distribution —
    # this is the (1) class-imbalance half of "balanced training" (see losses.py).
    logger.info("Computing class weights over %d training circuits...", len(train_rows))
    all_labels = [label_horizontal_actions_fixed_point(r["raw_qasm"]) for r in tqdm(train_rows, desc="Labeling")]
    class_weights = compute_stage1_class_weights(all_labels)
    logger.info("Stage 1 class weights (KEEP, CANCEL, MERGE_INTO_PREV): %s", class_weights.tolist())

    model = Stage1HorizontalModel().to(device)
    criterion = Stage1BalancedLoss(class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_pass_rate = -1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        random.shuffle(train_rows)

        for row in tqdm(train_rows, desc=f"Epoch {epoch + 1}/{epochs}"):
            _, graph, action_targets, angle_targets, angle_mask = build_example(row)
            graph.data = graph.data.to(device)
            action_targets, angle_targets, angle_mask = (
                action_targets.to(device), angle_targets.to(device), angle_mask.to(device),
            )

            optimizer.zero_grad()
            action_logits, predicted_angle = model(graph.data)
            losses = criterion(action_logits, predicted_angle, action_targets, angle_targets, angle_mask)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += losses["total"].item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_rows), 1)

        # Checksum-gated validation — the (2) compression-vs-fidelity balance.
        pass_rate = evaluate_checksum_pass_rate(model, val_rows, device)
        logger.info("Epoch %d: train_loss=%.4f  checksum1_pass_rate=%.3f", epoch + 1, avg_loss, pass_rate)

        if pass_rate >= best_pass_rate:
            best_pass_rate = pass_rate
            torch.save({"model_state": model.state_dict(), "class_weights": class_weights, "pass_rate": pass_rate}, out_path)
            logger.info("  -> new best checkpoint saved (pass_rate=%.3f)", pass_rate)

    logger.info("Training complete. Best Checksum-1 pass rate: %.3f. Checkpoint: %s", best_pass_rate, out_path)


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
