"""SPARTA biochemical and physics-based transformer modules."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("Softplus-constrained initial values must be positive.")
    return math.log(math.expm1(value))


class SqueezeExcitationBlock(nn.Module):
    """Squeeze-and-excitation recalibration for one-dimensional features."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.network = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _ = inputs.shape
        weights = self.average_pool(inputs).view(batch_size, channels)
        weights = self.network(weights).view(batch_size, channels, 1)
        return inputs * weights


class ResidualConvolutionBlock(nn.Module):
    """Residual convolution block with channel-wise recalibration."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.squeeze_excitation = SqueezeExcitationBlock(channels)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.squeeze_excitation(self.network(inputs))
        return self.activation(inputs + residual)


class BiochemicalConvolutionTower(nn.Module):
    """Pool, residual convolution, pool, and output projection."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.MaxPool1d(2),
            ResidualConvolutionBlock(
                input_channels,
                kernel_size=5,
                dropout=dropout,
            ),
            nn.MaxPool1d(2),
            nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class RegulatoryNodeEncoder(nn.Module):
    """Split-stem biochemical encoder for DNA and epigenomic signals."""

    def __init__(
        self,
        sequence_channels: int = 4,
        epigenomic_channels: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        stem_channels = 32
        self.sequence_stem = nn.Sequential(
            nn.Conv1d(sequence_channels, stem_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
        )
        self.epigenomic_stem = nn.Sequential(
            nn.Conv1d(epigenomic_channels, stem_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
        )
        self.tower = BiochemicalConvolutionTower(
            input_channels=stem_channels * 2,
            output_channels=128,
            dropout=dropout,
        )
        self.projection = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _global_pool(features: torch.Tensor) -> torch.Tensor:
        maximum = F.adaptive_max_pool1d(features, 1).squeeze(-1)
        average = F.adaptive_avg_pool1d(features, 1).squeeze(-1)
        return torch.cat([maximum, average], dim=1)

    def forward(
        self,
        dna_sequence: torch.Tensor,
        epigenomic_signals: torch.Tensor,
    ) -> torch.Tensor:
        if dna_sequence.ndim != 4 or epigenomic_signals.ndim != 4:
            raise ValueError("Expected DNA and epigenomic tensors with shapes [batch, nodes, channels, window].")
        batch_size, number_of_nodes, sequence_channels, window_bp = dna_sequence.shape
        epi_batch, epi_nodes, epigenomic_channels, epi_window = epigenomic_signals.shape
        if (batch_size, number_of_nodes, window_bp) != (
            epi_batch,
            epi_nodes,
            epi_window,
        ):
            raise ValueError("DNA and epigenomic tensors have incompatible dimensions.")

        sequence_inputs = dna_sequence.reshape(
            batch_size * number_of_nodes,
            sequence_channels,
            window_bp,
        )
        epigenomic_inputs = epigenomic_signals.reshape(
            batch_size * number_of_nodes,
            epigenomic_channels,
            window_bp,
        )
        sequence_features = self.sequence_stem(sequence_inputs)
        epigenomic_features = self.epigenomic_stem(epigenomic_inputs)
        features = self.tower(torch.cat([sequence_features, epigenomic_features], dim=1))
        pooled = self._global_pool(features)
        embeddings = self.projection(pooled)
        return embeddings.view(batch_size, number_of_nodes, -1)


class RegulatoryPotentialHead(nn.Module):
    """Estimate an unconstrained activation or repressive potential per node."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(node_embeddings).squeeze(-1)


class BiochemicalAffinityHead(nn.Module):
    """Estimate TSS-to-node biochemical affinity in the interval [0, 1]."""

    def __init__(self, input_dim: int = 256) -> None:
        super().__init__()
        self.scale = input_dim**-0.5
        self.promoter_projection = nn.Linear(input_dim, input_dim, bias=False)
        self.node_projection = nn.Linear(input_dim, input_dim, bias=False)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        promoter_indices: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = node_embeddings.shape[0]
        batch_indices = torch.arange(batch_size, device=node_embeddings.device)
        promoter_embeddings = node_embeddings[batch_indices, promoter_indices]
        queries = self.promoter_projection(promoter_embeddings)
        keys = self.node_projection(node_embeddings)
        scores = (keys * queries[:, None, :]).sum(dim=-1) * self.scale
        affinity = torch.sigmoid(scores)
        if node_mask is not None:
            affinity = affinity * node_mask.to(affinity.dtype)
        return affinity


class BasalExpressionHead(nn.Module):
    """Estimate distance-independent basal expression from the TSS embedding."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, promoter_embedding: torch.Tensor) -> torch.Tensor:
        return self.network(promoter_embedding).squeeze(-1)


class PathInsulationHead(nn.Module):
    """Estimate cumulative structural resistance between every pair of nodes."""

    def __init__(
        self,
        input_dim: int = 256,
        projection_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, projection_dim)
        self.resistance_network = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.resistance_network[-1].bias, -3.0)

    def individual_score(self, path_insulation_embedding: torch.Tensor) -> torch.Tensor:
        """Apply the trained resistance network to individual node embeddings."""

        return F.softplus(self.resistance_network(path_insulation_embedding).squeeze(-1))

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, number_of_nodes, _ = node_embeddings.shape
        path_embeddings = self.projection(node_embeddings)
        if node_mask is not None:
            path_embeddings = path_embeddings * node_mask[:, :, None].to(path_embeddings.dtype)

        cumulative = torch.cumsum(path_embeddings, dim=1)
        cumulative_before = torch.cat(
            [torch.zeros_like(path_embeddings[:, :1]), cumulative[:, :-1]],
            dim=1,
        )
        path_sum = cumulative_before.unsqueeze(1) - cumulative.unsqueeze(2)
        resistance = F.softplus(self.resistance_network(path_sum).squeeze(-1))

        has_intermediate_node = torch.triu(
            torch.ones(
                number_of_nodes,
                number_of_nodes,
                dtype=torch.bool,
                device=node_embeddings.device,
            ),
            diagonal=2,
        )
        resistance = resistance * has_intermediate_node
        resistance = resistance + resistance.transpose(1, 2)
        if node_mask is not None:
            pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
            resistance = resistance * pair_mask.to(resistance.dtype)
        if resistance.shape != (batch_size, number_of_nodes, number_of_nodes):
            raise RuntimeError("Unexpected path-insulation force shape.")
        return resistance, path_embeddings


def pairwise_genomic_distance(genomic_coordinates: torch.Tensor) -> torch.Tensor:
    """Calculate absolute pairwise genomic distances."""

    return (genomic_coordinates[:, :, None].float() - genomic_coordinates[:, None, :].float()).abs()


class PhysicsBasedTransformerLayer(nn.Module):
    """One tri-force attention layer constrained by polymer-physics terms."""

    def __init__(
        self,
        hidden_dim: int,
        number_of_heads: int,
        dropout: float = 0.1,
        initial_contact_energy: float = -3.0,
    ) -> None:
        super().__init__()
        if hidden_dim % number_of_heads:
            raise ValueError("hidden_dim must be divisible by number_of_heads.")
        self.hidden_dim = hidden_dim
        self.number_of_heads = number_of_heads
        self.head_dim = hidden_dim // number_of_heads
        self.scale = self.head_dim**-0.5

        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)

        initial_weight = _inverse_softplus(1.0)
        self.raw_loop_weight = nn.Parameter(torch.tensor(initial_weight))
        self.raw_distance_weight = nn.Parameter(torch.tensor(initial_weight))
        self.raw_insulation_weight = nn.Parameter(torch.tensor(initial_weight))
        self.base_contact_energy = nn.Parameter(torch.tensor(initial_contact_energy))

    def forward(
        self,
        node_embeddings: torch.Tensor,
        distance_decay_force: torch.Tensor,
        path_insulation_force: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        return_energy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, number_of_nodes, hidden_dim = node_embeddings.shape
        residual = node_embeddings
        normalized = self.attention_norm(node_embeddings)

        def project(layer: nn.Linear) -> torch.Tensor:
            return (
                layer(normalized)
                .view(
                    batch_size,
                    number_of_nodes,
                    self.number_of_heads,
                    self.head_dim,
                )
                .transpose(1, 2)
            )

        queries = project(self.query_projection)
        keys = project(self.key_projection)
        values = project(self.value_projection)
        loop_force = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        loop_weight = F.softplus(self.raw_loop_weight)
        distance_weight = F.softplus(self.raw_distance_weight)
        insulation_weight = F.softplus(self.raw_insulation_weight)
        interaction_energy = (
            loop_weight * loop_force
            + distance_weight * distance_decay_force[:, None]
            - insulation_weight * path_insulation_force[:, None]
            + self.base_contact_energy
        )
        if node_mask is not None:
            key_mask = node_mask[:, None, None, :]
            interaction_energy = interaction_energy.masked_fill(~key_mask, -1e9)

        attention = F.softmax(interaction_energy, dim=-1)
        attention = self.attention_dropout(attention)
        output = torch.matmul(attention, values)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                number_of_nodes,
                hidden_dim,
            )
        )
        output = residual + self.output_dropout(self.output_projection(output))
        output = output + self.feed_forward(self.output_norm(output))
        if node_mask is not None:
            output = output * node_mask[:, :, None].to(output.dtype)

        if return_energy:
            return output, interaction_energy
        return output


class PhysicsBasedTransformer(nn.Module):
    """Reconstruct chromatin interaction energy with tri-force attention."""

    def __init__(
        self,
        hidden_dim: int = 256,
        number_of_layers: int = 4,
        number_of_heads: int = 8,
        dropout: float = 0.1,
        initial_distance_reference_bp: float = 10000.0,
        initial_distance_decay: float = 1.0,
        insulation_projection_dim: int = 32,
    ) -> None:
        super().__init__()
        if initial_distance_reference_bp <= 0:
            raise ValueError("initial_distance_reference_bp must be positive.")
        self.path_insulation_head = PathInsulationHead(
            input_dim=hidden_dim,
            projection_dim=insulation_projection_dim,
            hidden_dim=64,
        )
        self.log_distance_reference = nn.Parameter(torch.tensor(math.log(initial_distance_reference_bp)))
        self.raw_distance_decay = nn.Parameter(torch.tensor(_inverse_softplus(initial_distance_decay)))
        self.layers = nn.ModuleList(
            [
                PhysicsBasedTransformerLayer(
                    hidden_dim,
                    number_of_heads,
                    dropout=dropout,
                )
                for _ in range(number_of_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.structural_head_index = 0

    def distance_decay_force(
        self,
        genomic_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the learnable polymer distance-decay relationship."""

        distance = pairwise_genomic_distance(genomic_coordinates)
        reference = torch.exp(self.log_distance_reference).clamp_min(1.0)
        exponent = F.softplus(self.raw_distance_decay).clamp_min(1e-3)
        return torch.exp(-exponent * torch.log1p(distance / reference))

    def forward(
        self,
        node_embeddings: torch.Tensor,
        genomic_coordinates: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        path_force, path_embeddings = self.path_insulation_head(
            node_embeddings,
            node_mask=node_mask,
        )
        distance_force = self.distance_decay_force(genomic_coordinates)
        if node_mask is not None:
            pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
            distance_force = distance_force * pair_mask.to(distance_force.dtype)

        transformed = node_embeddings
        if node_mask is not None:
            transformed = transformed * node_mask[:, :, None].to(transformed.dtype)
        final_multihead_energy: torch.Tensor | None = None
        for layer_index, layer in enumerate(self.layers):
            is_final_layer = layer_index == len(self.layers) - 1
            if is_final_layer:
                transformed, final_multihead_energy = layer(
                    transformed,
                    distance_force,
                    path_force,
                    node_mask=node_mask,
                    return_energy=True,
                )
            else:
                transformed = layer(
                    transformed,
                    distance_force,
                    path_force,
                    node_mask=node_mask,
                )
        transformed = self.final_norm(transformed)
        if node_mask is not None:
            transformed = transformed * node_mask[:, :, None].to(transformed.dtype)

        if final_multihead_energy is None:
            raise RuntimeError("The physics-based transformer must contain at least one layer.")
        interaction_energy = final_multihead_energy[:, self.structural_head_index]
        interaction_energy = 0.5 * (interaction_energy + interaction_energy.transpose(1, 2))
        if node_mask is not None:
            pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
            interaction_energy = interaction_energy * pair_mask.to(interaction_energy.dtype)
        return {
            "physics_embedding": transformed,
            "interaction_energy": interaction_energy,
            "distance_decay_force": distance_force,
            "path_insulation_force": path_force,
            "path_insulation_embedding": path_embeddings,
        }


class SpartaModel(nn.Module):
    """Complete SPARTA model with an interpretable linear additive readout."""

    def __init__(
        self,
        sequence_channels: int = 4,
        epigenomic_channels: int = 8,
        hidden_dim: int = 256,
        physics_layers: int = 4,
        physics_heads: int = 8,
        dropout: float = 0.1,
        insulation_projection_dim: int = 32,
        initial_distance_reference_bp: float = 10000.0,
        initial_distance_decay: float = 1.0,
        initial_temperature: float = 1.0,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive.")
        self.regulatory_node_encoder = RegulatoryNodeEncoder(
            sequence_channels=sequence_channels,
            epigenomic_channels=epigenomic_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.regulatory_potential_head = RegulatoryPotentialHead(
            input_dim=hidden_dim,
            hidden_dim=128,
            dropout=dropout,
        )
        self.biochemical_affinity_head = BiochemicalAffinityHead(input_dim=hidden_dim)
        self.basal_expression_head = BasalExpressionHead(
            input_dim=hidden_dim,
            hidden_dim=64,
            dropout=dropout,
        )
        self.physics_based_transformer = PhysicsBasedTransformer(
            hidden_dim=hidden_dim,
            number_of_layers=physics_layers,
            number_of_heads=physics_heads,
            dropout=dropout,
            initial_distance_reference_bp=initial_distance_reference_bp,
            initial_distance_decay=initial_distance_decay,
            insulation_projection_dim=insulation_projection_dim,
        )
        temperature = torch.tensor(math.log(initial_temperature))
        if learnable_temperature:
            self.log_interaction_temperature = nn.Parameter(temperature)
        else:
            self.register_buffer(
                "log_interaction_temperature",
                temperature,
                persistent=True,
            )

    @staticmethod
    def _promoter_indices(
        promoter_mask: torch.Tensor,
        node_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid_promoter = promoter_mask & node_mask if node_mask is not None else promoter_mask
        promoter_counts = valid_promoter.sum(dim=1)
        if not torch.all(promoter_counts == 1):
            raise ValueError("Each sample must contain exactly one valid promoter node.")
        return valid_promoter.to(torch.int64).argmax(dim=1)

    def forward(
        self,
        dna_sequence: torch.Tensor,
        epigenomic_signals: torch.Tensor,
        genomic_coordinates: torch.Tensor,
        promoter_mask: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        promoter_indices = self._promoter_indices(promoter_mask, node_mask)
        node_embeddings = self.regulatory_node_encoder(
            dna_sequence,
            epigenomic_signals,
        )
        if node_mask is not None:
            node_embeddings = node_embeddings * node_mask[:, :, None].to(node_embeddings.dtype)

        regulatory_potential = self.regulatory_potential_head(node_embeddings)
        if node_mask is not None:
            regulatory_potential = regulatory_potential * node_mask.to(regulatory_potential.dtype)
        biochemical_affinity = self.biochemical_affinity_head(
            node_embeddings,
            promoter_indices,
            node_mask=node_mask,
        )
        batch_size = node_embeddings.shape[0]
        batch_indices = torch.arange(batch_size, device=node_embeddings.device)
        promoter_embeddings = node_embeddings[batch_indices, promoter_indices]
        basal_expression = self.basal_expression_head(promoter_embeddings)

        physics_outputs = self.physics_based_transformer(
            node_embeddings,
            genomic_coordinates,
            node_mask=node_mask,
        )
        interaction_energy = physics_outputs["interaction_energy"]
        temperature = torch.exp(self.log_interaction_temperature).clamp_min(1e-3)
        interaction_probability = torch.sigmoid(interaction_energy / temperature)
        if node_mask is not None:
            pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
            interaction_probability = interaction_probability * pair_mask.to(interaction_probability.dtype)
        promoter_interaction_probability = interaction_probability[
            batch_indices,
            promoter_indices,
            :,
        ]

        regulatory_effect = regulatory_potential * biochemical_affinity * promoter_interaction_probability
        regulatory_effect[batch_indices, promoter_indices] = 0.0
        predicted_expression = basal_expression + regulatory_effect.sum(dim=1)

        outputs: dict[str, torch.Tensor | None] = {
            "predicted_expression": predicted_expression,
            "basal_expression": basal_expression,
            "regulatory_potential": regulatory_potential,
            "biochemical_affinity": biochemical_affinity,
            "promoter_interaction_probability": promoter_interaction_probability,
            "regulatory_effect": regulatory_effect,
            "interaction_temperature": temperature,
            "interaction_energy": interaction_energy if return_intermediates else None,
        }
        if return_intermediates:
            outputs.update(
                {
                    "interaction_probability": interaction_probability,
                    "distance_decay_force": physics_outputs["distance_decay_force"],
                    "path_insulation_force": physics_outputs["path_insulation_force"],
                    "path_insulation_embedding": physics_outputs["path_insulation_embedding"],
                    "regulatory_node_embedding": node_embeddings,
                    "physics_embedding": physics_outputs["physics_embedding"],
                }
            )
        return outputs

    @classmethod
    def from_config(
        cls,
        model_config: dict[str, Any],
        epigenomic_channels: int,
    ) -> SpartaModel:
        """Construct a model from the `model` YAML section."""

        return cls(
            sequence_channels=int(model_config.get("sequence_channels", 4)),
            epigenomic_channels=epigenomic_channels,
            hidden_dim=int(model_config["hidden_dim"]),
            physics_layers=int(model_config["physics_layers"]),
            physics_heads=int(model_config["physics_heads"]),
            dropout=float(model_config.get("dropout", 0.1)),
            insulation_projection_dim=int(model_config.get("insulation_projection_dim", 32)),
            initial_distance_reference_bp=float(model_config.get("initial_distance_reference_bp", 10000.0)),
            initial_distance_decay=float(model_config.get("initial_distance_decay", 1.0)),
            initial_temperature=float(model_config.get("initial_temperature", 1.0)),
            learnable_temperature=bool(model_config.get("learnable_temperature", True)),
        )
