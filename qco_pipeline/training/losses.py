"""
"Balanced" training objectives.

"Balanced" is addressed on two separate axes, both real failure modes if
ignored:

  (1) CLASS-IMBALANCE balance. In a typical circuit, KEEP vastly outnumbers
      CANCEL/MERGE_INTO_PREV (most gates aren't part of a cancelling pair),
      and most wires are KEEP, not prune. An unweighted loss lets the model
      collapse to "always predict the majority class" and still score high
      accuracy while doing zero useful optimization. We counter this with
      inverse-frequency class weighting (Stage 1) and pos_weight (Stage 2).

  (2) COMPRESSION-vs-FIDELITY balance. A model can trivially get 100%
      "compression" by deleting everything, or 100% "safety" by deleting
      nothing. Since fidelity is computed via circuit simulation (Checksum 1
      / Checksum 2) and is not differentiable through the model's discrete
      argmax decisions, we do not fold it into the backprop loss directly.
      Instead: (a) the supervised targets themselves are ALREADY fidelity-
      exact (Phase 0's classical-compiler / rule-based labels never break
      semantics — see phase0/horizontal_labels.py), so a model that fits
      the labels well is fidelity-safe by construction; (b) we still track
      checksum PASS RATE as a held-out validation metric every epoch, and
      use it (not just loss) for checkpoint selection and early stopping —
      this is what actually keeps training "balanced" between the two
      failure modes in practice, not a hand-tuned differentiable penalty.
"""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from qco_pipeline.phase0.horizontal_labels import Action


def compute_stage1_class_weights(all_labels: list[list["Action"]]) -> torch.Tensor:
    """Inverse-frequency class weights over the 3-way Stage 1 action label."""
    counts = Counter()
    for labels in all_labels:
        counts.update(int(lbl.action) for lbl in labels)

    total = sum(counts.values()) or 1
    weights = torch.ones(len(Action))
    for action_id in range(len(Action)):
        freq = counts.get(action_id, 0) / total
        weights[action_id] = 1.0 / freq if freq > 0 else 0.0

    # Normalize so the weighted CE stays on a comparable scale to unweighted CE.
    weights = weights / weights.sum() * len(Action)
    return weights


def compute_stage2_pos_weight(all_keep_masks: list[torch.Tensor]) -> torch.Tensor:
    """
    pos_weight for BCEWithLogitsLoss, where "positive" = prune.
    If prune events are rare (typical), pos_weight > 1 upweights them so the
    model doesn't just predict "keep everything" and call it done.
    """
    total_wires = sum(mask.numel() for mask in all_keep_masks)
    total_pruned = sum((~mask).sum().item() for mask in all_keep_masks)
    total_kept = total_wires - total_pruned

    if total_pruned == 0:
        return torch.tensor(1.0)
    return torch.tensor(total_kept / total_pruned)


class Stage1BalancedLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, angle_loss_weight: float = 0.5):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.angle_loss_weight = angle_loss_weight

    def forward(
        self,
        action_logits: torch.Tensor,      # (num_nodes, 3)
        predicted_angle: torch.Tensor,    # (num_nodes,)
        action_targets: torch.Tensor,     # (num_nodes,) long
        angle_targets: torch.Tensor,      # (num_nodes,) float
        angle_mask: torch.Tensor,         # (num_nodes,) bool — which nodes have a meaningful angle target
    ) -> dict[str, torch.Tensor]:
        action_loss = F.cross_entropy(action_logits, action_targets, weight=self.class_weights.to(action_logits.device))

        if angle_mask.any():
            angle_loss = F.mse_loss(predicted_angle[angle_mask], angle_targets[angle_mask])
        else:
            angle_loss = torch.tensor(0.0, device=action_logits.device)

        total = action_loss + self.angle_loss_weight * angle_loss
        return {"total": total, "action_loss": action_loss, "angle_loss": angle_loss}


class Stage2BalancedLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, prune_logit: torch.Tensor, keep_mask_target: torch.Tensor) -> dict[str, torch.Tensor]:
        prune_target = (~keep_mask_target).float()  # BCE target: 1 = prune
        loss = F.binary_cross_entropy_with_logits(prune_logit, prune_target, pos_weight=self.pos_weight.to(prune_logit.device))
        return {"total": loss}
