"""Shape and terminology smoke tests for the SPARTA model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sparta.models import SpartaModel  # noqa: E402


def test_model_outputs_paper_aligned_terms() -> None:
    model = SpartaModel(
        epigenomic_channels=2,
        hidden_dim=32,
        physics_layers=2,
        physics_heads=4,
        insulation_projection_dim=8,
        dropout=0.0,
    ).eval()
    batch_size = 2
    number_of_nodes = 4
    window_bp = 32
    dna = torch.zeros(batch_size, number_of_nodes, 4, window_bp)
    dna[:, :, 0] = 1.0
    epigenomic = torch.rand(batch_size, number_of_nodes, 2, window_bp)
    coordinates = torch.tensor([[-1000.0, 0.0, 2000.0, 0.0], [-2000.0, 0.0, 3000.0, 0.0]])
    promoter_mask = torch.tensor([[False, True, False, False], [False, True, False, False]])
    node_mask = torch.tensor([[True, True, True, False], [True, True, True, False]])

    with torch.no_grad():
        outputs = model(
            dna,
            epigenomic,
            coordinates,
            promoter_mask,
            node_mask,
            return_intermediates=True,
        )

    assert outputs["predicted_expression"].shape == (batch_size,)
    assert outputs["regulatory_potential"].shape == (batch_size, number_of_nodes)
    assert outputs["biochemical_affinity"].shape == (batch_size, number_of_nodes)
    assert outputs["interaction_energy"].shape == (
        batch_size,
        number_of_nodes,
        number_of_nodes,
    )
    assert outputs["path_insulation_embedding"].shape[:2] == (
        batch_size,
        number_of_nodes,
    )
    assert torch.all(outputs["regulatory_effect"][:, 1] == 0)
    assert torch.all(outputs["regulatory_effect"][:, 3] == 0)
