"""
modelss/encoders/tabular.py

Strict 3-layer MLP tabular encoder: input_dim → 64 → 32 → 16.

Architecture (NO BatchNorm, NO Dropout per specification)
---------------------------------------------------------
  nn.Linear(input_dim, 64) → nn.ReLU()
  nn.Linear(64, 32)        → nn.ReLU()
  nn.Linear(32, 16)
      ↓
  [N, 16]

The fixed 16-dim output is a hard architectural constant.  The fusion layer
depends on receiving exactly 16 dimensions from the tabular branch.

``input_dim`` is the number of columns produced by the upstream
``ColumnTransformer`` (StandardScaler + median imputation).  It must be
set at construction time from the Phase 3 preprocessor's output shape.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Architectural constant – must match the fusion-layer dimension contract
TABULAR_OUTPUT_DIM: int = 16


class TabularEncoder(nn.Module):
    """
    Strict 3-layer MLP encoder for preprocessed tabular feature vectors.

    Parameters
    ----------
    input_dim : int
        Number of features output by the upstream ``ColumnTransformer``.
        Typically determined at Phase 3 preprocessing time.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()

        self.input_dim: int = input_dim

        # ── 3-layer MLP (strict spec – NO BatchNorm, NO Dropout) ─────────
        self.network: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, TABULAR_OUTPUT_DIM),
        )

        logger.info(
            "TabularEncoder: input_dim=%d  topology=%d→64→32→%d  "
            "output_dim=%d",
            input_dim, input_dim, TABULAR_OUTPUT_DIM, TABULAR_OUTPUT_DIM,
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of tabular feature vectors.

        Parameters
        ----------
        x : torch.Tensor
            Float tensor of shape ``(N, input_dim)`` — scaled tabular
            features produced by the Phase 3 ``ColumnTransformer``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 16)`` — encoded tabular embeddings.
        """
        return self.network(x)

    def get_output_dim(self) -> int:
        """Return the fixed output dimensionality (16)."""
        return TABULAR_OUTPUT_DIM
