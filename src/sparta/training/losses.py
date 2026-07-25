"""Loss functions used by the SPARTA two-stage training strategy."""

from __future__ import annotations

import math
from itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F


def pearson_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Calculate a numerically stable Pearson correlation coefficient."""

    prediction = prediction.flatten()
    target = target.flatten()
    if prediction.numel() != target.numel():
        raise ValueError("Pearson inputs must contain the same number of values.")
    if prediction.numel() == 0:
        return prediction.sum() * 0.0
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    numerator = (centered_prediction * centered_target).sum()
    denominator = torch.sqrt(centered_prediction.square().sum() * centered_target.square().sum() + epsilon)
    return numerator / denominator


class StratifiedHiCPearsonLoss(nn.Module):
    """Distance-stratified Pearson loss against observed-over-expected Hi-C."""

    def __init__(
        self,
        distance_bins_bp: list[int] | tuple[int, ...] = (20000, 100000),
        minimum_pairs_per_bin: int = 30,
    ) -> None:
        super().__init__()
        self.distance_bins_bp = list(distance_bins_bp)
        self.minimum_pairs_per_bin = int(minimum_pairs_per_bin)

    def forward(
        self,
        interaction_energy: torch.Tensor,
        hic_oe: torch.Tensor,
        genomic_coordinates: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        _, number_of_nodes, _ = interaction_energy.shape
        distances = (genomic_coordinates[:, :, None].float() - genomic_coordinates[:, None, :].float()).abs()
        upper_triangle = torch.triu(
            torch.ones(
                number_of_nodes,
                number_of_nodes,
                dtype=torch.bool,
                device=interaction_energy.device,
            ),
            diagonal=1,
        )
        valid = node_mask[:, :, None] & node_mask[:, None, :] & upper_triangle
        predictions = interaction_energy[valid]
        targets = torch.log1p(hic_oe[valid].clamp_min(0.0))
        pair_distances = distances[valid]
        if predictions.numel() == 0:
            return interaction_energy.sum() * 0.0, {}

        boundaries = [0, *self.distance_bins_bp, math.inf]
        names = ("short", "mid", "long")
        correlations: list[torch.Tensor] = []
        weights: list[float] = []
        metrics: dict[str, float] = {}
        for index, (lower, upper) in enumerate(pairwise(boundaries)):
            selected = (pair_distances >= lower) & (pair_distances < upper)
            count = int(selected.sum().item())
            if count < self.minimum_pairs_per_bin:
                continue
            correlation = pearson_correlation(
                predictions[selected],
                targets[selected],
            )
            correlations.append(correlation)
            weights.append(math.log1p(count))
            label = names[index] if index < len(names) else f"bin_{index}"
            metrics[f"stratified_{label}_hic_r"] = float(correlation.detach().item())

        if not correlations:
            return interaction_energy.sum() * 0.0, metrics
        weight_tensor = torch.tensor(
            weights,
            dtype=interaction_energy.dtype,
            device=interaction_energy.device,
        )
        mean_correlation = (torch.stack(correlations) * weight_tensor).sum() / weight_tensor.sum()
        metrics["stratified_mean_hic_r"] = float(mean_correlation.detach().item())
        return 1.0 - mean_correlation, metrics


class GlobalHiCPearsonLoss(nn.Module):
    """Global Pearson loss against SCALE-normalized Hi-C contacts."""

    def __init__(self, minimum_pairs: int = 100) -> None:
        super().__init__()
        self.minimum_pairs = int(minimum_pairs)

    def forward(
        self,
        interaction_energy: torch.Tensor,
        hic_scale: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        _, number_of_nodes, _ = interaction_energy.shape
        upper_triangle = torch.triu(
            torch.ones(
                number_of_nodes,
                number_of_nodes,
                dtype=torch.bool,
                device=interaction_energy.device,
            ),
            diagonal=1,
        )
        valid = node_mask[:, :, None] & node_mask[:, None, :] & upper_triangle
        predictions = interaction_energy[valid]
        if predictions.numel() < self.minimum_pairs:
            return interaction_energy.sum() * 0.0, {"global_hic_r": 0.0}
        targets = torch.log1p(hic_scale[valid].clamp_min(0.0))
        correlation = pearson_correlation(predictions, targets)
        return 1.0 - correlation, {"global_hic_r": float(correlation.detach().item())}


class ExpressionPredictionLoss(nn.Module):
    """Smooth L1 expression loss with a batch-level Pearson metric."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prediction = prediction.flatten()
        target = target.flatten()
        loss = F.smooth_l1_loss(prediction, target)
        with torch.no_grad():
            correlation = pearson_correlation(prediction, target)
        return loss, {
            "expression_smooth_l1": float(loss.detach().item()),
            "expression_r": float(correlation.item()),
        }
