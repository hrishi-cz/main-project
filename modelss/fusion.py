"""
modelss/fusion.py

Fusion strategies for multimodal encoder outputs.

Both classes expose a ``get_output_dim() -> int`` method so downstream
layers (``MultimodalPredictor``) can query the output dimensionality without
instantiating any tensors.

Dimension contract (all three modalities active)
------------------------------------------------
  ImageEncoder   → [N, 512]   (ResNet-50 → Linear(2048, 512) → ReLU)
  TextEncoder    → [N, 768]   (BERT CLS token)
  TabularEncoder → [N,  16]   (MLP input→64→32→16)
                         ↓
  ConcatenationFusion output: 512 + 768 + 16 = 1296 dims

When fewer than three modalities are active, ``MultimodalPredictor``
passes ``torch.full(shape, 1e-7)`` dummy tensors for the absent
modalities so the concatenated dimension is always 1296.

ConcatenationFusion
-------------------
Dynamically computes its ``output_dim`` as ``sum(feature_dims)`` at
construction time.  No learnable parameters — the output is the raw
horizontal concatenation of all modality tensors along ``dim=1``.

AttentionFusion
---------------
Projects every modality tensor to a shared ``latent_dim`` (default 512)
with independent ``nn.Linear`` layers, then scores each projected
embedding with a shared attention network:

    nn.Sequential(
        nn.Linear(latent_dim, latent_dim),
        nn.Tanh(),
        nn.Linear(latent_dim, 1),
    )

Scores are softmax-normalised across the modality axis and used to
compute a single attention-weighted context vector of shape
``(N, latent_dim)``.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConcatenationFusion
# ---------------------------------------------------------------------------

class ConcatenationFusion(nn.Module):
    """
    Horizontal concatenation of modality embedding tensors.

    ``output_dim`` is determined dynamically as ``sum(feature_dims)`` so
    it always tracks the exact input dimensionality regardless of how many
    encoders are active.

    Parameters
    ----------
    feature_dims : List[int]
        Output dimensionality of each active encoder, in the same order
        that tensors will be passed to ``forward()``.
        Example: ``[512, 768, 16]`` for ResNet-50 + BERT-base + Tabular-MLP.
    """

    def __init__(self, feature_dims: List[int]) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError(
                "ConcatenationFusion requires at least one encoder dimension."
            )
        self.feature_dims: List[int] = feature_dims
        self._output_dim: int = sum(feature_dims)
        logger.info(
            "ConcatenationFusion: %d modalities  dims=%s  output_dim=%d",
            len(feature_dims), feature_dims, self._output_dim,
        )

    # ------------------------------------------------------------------ #

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate modality tensors along the feature axis.

        Parameters
        ----------
        features : List[torch.Tensor]
            One tensor per active modality, each of shape ``(N, d_i)``.
            Must be non-empty; length must equal ``len(feature_dims)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, sum(d_i))`` — the horizontally concatenated tensor.
        """
        if not features:
            raise ValueError(
                "ConcatenationFusion.forward received an empty feature list."
            )
        return torch.cat(features, dim=1)

    def get_output_dim(self) -> int:
        """Return the output feature dimensionality (== ``sum(feature_dims)``)."""
        return self._output_dim


# ---------------------------------------------------------------------------
# AttentionFusion
# ---------------------------------------------------------------------------

class AttentionFusion(nn.Module):
    """
    Attention-weighted fusion of multimodal encoder outputs.

    Each modality tensor is first projected to a shared ``latent_dim``
    (default 512) by an independent ``nn.Linear`` layer.  A single shared
    attention-scoring network then assigns a scalar importance weight to
    each projected embedding:

        score(e) = Linear(latent_dim → 1)(Tanh(Linear(latent_dim → latent_dim)(e)))

    Weights are softmax-normalised across the modality axis and used to
    compute an attention-weighted sum — the context vector of shape
    ``(N, latent_dim)``.

    Parameters
    ----------
    feature_dims : List[int]
        Output dimensionality of each active encoder, in the same order
        that tensors will be passed to ``forward()``.
    latent_dim : int
        Shared projection dimensionality (default 512).  All modality
        projections land in this space before attention scoring.
    """

    def __init__(
        self,
        feature_dims: List[int],
        latent_dim: int = 512,
    ) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError(
                "AttentionFusion requires at least one encoder dimension."
            )
        self.feature_dims: List[int] = feature_dims
        self._latent_dim: int = latent_dim

        # ── Per-modality projection layers ────────────────────────────────
        # Each encoder's output is mapped to the shared latent space
        # independently so that dimension mismatches across modalities are
        # resolved before attention scoring.
        self.projections: nn.ModuleList = nn.ModuleList([
            nn.Linear(d, latent_dim) for d in feature_dims
        ])

        # ── Shared attention-scoring network ──────────────────────────────
        # Applied to every projected embedding (shape: (N, n_mod, latent_dim))
        # via PyTorch's implicit last-dim broadcast.
        #
        #   Linear(latent_dim, latent_dim) → Tanh → Linear(latent_dim, 1)
        #
        # Output shape: (N, n_mod, 1) — one scalar score per modality.
        self.attention_scoring: nn.Sequential = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1),
        )

        logger.info(
            "AttentionFusion: %d modalities  dims=%s  latent_dim=%d",
            len(feature_dims), feature_dims, latent_dim,
        )

    # ------------------------------------------------------------------ #

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute attention-weighted context vector from modality tensors.

        Parameters
        ----------
        features : List[torch.Tensor]
            One tensor per active modality, each of shape ``(N, d_i)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, latent_dim)`` — the attention-weighted sum of the
            projected modality embeddings (context vector).
        """
        if not features:
            raise ValueError(
                "AttentionFusion.forward received an empty feature list."
            )

        # 1. Project each modality to shared latent space
        #    projected_i shape: (N, latent_dim)
        projected: List[torch.Tensor] = [
            proj(feat) for proj, feat in zip(self.projections, features)
        ]

        # 2. Stack along a new modality axis
        #    stacked shape: (N, n_modalities, latent_dim)
        stacked: torch.Tensor = torch.stack(projected, dim=1)

        # 3. Score each projected embedding
        #    nn.Linear operates on the last dim, so it broadcasts across N
        #    and n_modalities automatically.
        #    scores shape: (N, n_modalities, 1)
        scores: torch.Tensor = self.attention_scoring(stacked)

        # 4. Softmax across the modality axis → normalised importance weights
        #    weights shape: (N, n_modalities, 1)
        weights: torch.Tensor = torch.softmax(scores, dim=1)

        # 5. Weighted sum collapses the modality axis
        #    context shape: (N, latent_dim)
        context: torch.Tensor = (stacked * weights).sum(dim=1)
        return context

    def get_output_dim(self) -> int:
        """Return the output feature dimensionality (== ``latent_dim``)."""
        return self._latent_dim
