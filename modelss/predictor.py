"""
modelss/predictor.py

MultimodalPredictor – full specification architecture.

Data flow
---------
  [ImageEncoder]   → image_features   (N, 512)   ┐
  [TextEncoder]    → text_features    (N, 768)   ├→ ConcatenationFusion → (N, 1296)
  [TabularEncoder] → tabular_features (N,  16)   ┘
                                                        ↓
                                              fusion_mlp (3 blocks / NO BatchNorm)
                                                        ↓
                                              predictor_head  (1 Linear + activation)
                                                        ↓
                                              output: (N, num_classes) or (N, 1)

Fusion MLP (NO BatchNorm anywhere)
----------
    Block 1: Linear(fusion_dim, 512) → ReLU() → Dropout(0.3)
    Block 2: Linear(512,         256) → ReLU() → Dropout(0.2)
    Block 3: Linear(256,         128) → ReLU()

Predictor Head
--------------
    classification : Linear(128, num_classes) → Softmax(dim=1)
    regression     : Linear(128, 1)           → Identity()

Missing-modality safety
-----------------------
If a modality's input tensor is ``None`` in ``forward()`` but the
corresponding encoder was registered at construction time, the absent
modality receives a ``torch.full((N, encoder_output_dim), 1e-7)`` dummy
tensor.  This preserves the concatenated dimension (e.g. 1296 when all
three encoders are active) and prevents NaN propagation from pure zeros.

The ``problem_type`` string (sourced from ``GlobalSchema.global_problem_type``)
drives which head variant is built at construction time.  No branching
occurs in ``forward()``.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional

import torch
import torch.nn as nn

from .encoders.image import ImageEncoder
from .encoders.tabular import TabularEncoder
from .encoders.text import TextEncoder
from .fusion import AttentionFusion, ConcatenationFusion

logger = logging.getLogger(__name__)


class MultimodalPredictor(nn.Module):
    """
    End-to-end multimodal prediction module.

    .. deprecated::
        This class is not used by the training pipeline (which uses
        ``automl.trainer._MultimodalHead`` instead).  It also bakes
        ``nn.Softmax`` into the forward pass, which is incompatible with
        ``nn.CrossEntropyLoss`` (double softmax).  Prefer
        ``_MultimodalHead`` for all new code.

    Accepts up to three modality-specific encoders; at least one must be
    provided.  All registered encoders always contribute to the fused
    representation — absent runtime inputs receive ``1e-7`` dummy tensors
    so the concatenated dimensionality is constant (1296 when all three
    encoders are active).

    Parameters
    ----------
    image_encoder : ImageEncoder | None
        ResNet-50 + projection head; outputs [N, 512].
    tabular_encoder : TabularEncoder | None
        Strict 3-layer MLP; outputs [N, 16].
    text_encoder : TextEncoder | None
        BERT / GPT-2 CLS encoder; outputs [N, 768].
    fusion_strategy : str
        ``"concatenation"`` (default) or ``"attention"``.
    problem_type : str
        One of ``"classification_binary"``, ``"classification_multiclass"``,
        or ``"regression"``.  Controls the output head.
    num_classes : int
        Number of target classes.  Ignored for regression (output_dim is
        fixed to 1).  Must be ≥ 2 for classification.
    """

    def __init__(
        self,
        image_encoder: Optional[ImageEncoder] = None,
        tabular_encoder: Optional[TabularEncoder] = None,
        text_encoder: Optional[TextEncoder] = None,
        fusion_strategy: str = "concatenation",
        problem_type: str = "classification_binary",
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        warnings.warn(
            "MultimodalPredictor is deprecated. Use automl.trainer._MultimodalHead "
            "instead. This class bakes nn.Softmax into forward(), which is "
            "incompatible with nn.CrossEntropyLoss.",
            DeprecationWarning,
            stacklevel=2,
        )

        # ── Encoder registration ──────────────────────────────────────────
        self.image_encoder: Optional[ImageEncoder]     = image_encoder
        self.tabular_encoder: Optional[TabularEncoder] = tabular_encoder
        self.text_encoder: Optional[TextEncoder]       = text_encoder
        self.problem_type: str = problem_type

        # ── Collect dimensions of ALL registered encoders ─────────────────
        # Dimensions are collected for every non-None encoder; dummy tensors
        # will be used at runtime if an input is missing.
        feature_dims: List[int] = []
        if image_encoder is not None:
            feature_dims.append(image_encoder.get_output_dim())     # 512
        if tabular_encoder is not None:
            feature_dims.append(tabular_encoder.get_output_dim())   # 16
        if text_encoder is not None:
            feature_dims.append(text_encoder.get_output_dim())      # 768

        if not feature_dims:
            raise ValueError(
                "MultimodalPredictor requires at least one active encoder "
                "(image_encoder, tabular_encoder, or text_encoder)."
            )

        # ── Fusion layer ──────────────────────────────────────────────────
        if fusion_strategy == "concatenation":
            self.fusion: nn.Module = ConcatenationFusion(feature_dims)
        elif fusion_strategy == "attention":
            self.fusion = AttentionFusion(feature_dims)
        else:
            raise ValueError(
                f"Unknown fusion_strategy '{fusion_strategy}'. "
                "Expected 'concatenation' or 'attention'."
            )

        fusion_dim: int = self.fusion.get_output_dim()

        # ── Fusion MLP (NO BatchNorm per spec) ────────────────────────────
        # Block 1: Linear(fusion_dim, 512) → ReLU → Dropout(0.3)
        # Block 2: Linear(512, 256)        → ReLU → Dropout(0.2)
        # Block 3: Linear(256, 128)        → ReLU
        self.fusion_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # ── Predictor head ────────────────────────────────────────────────
        # classification : Linear(128, num_classes) — raw logits (no activation)
        # regression     : Linear(128, 1)           → Identity()
        if problem_type.startswith("classification"):
            output_dim: int        = num_classes
            activation: nn.Module  = nn.Identity()
        else:
            output_dim  = 1
            activation  = nn.Identity()

        self.predictor_head: nn.Sequential = nn.Sequential(
            nn.Linear(128, output_dim),
            activation,
        )

        logger.info(
            "MultimodalPredictor: encoders=%s  fusion=%s  fusion_dim=%d  "
            "problem=%s  output_dim=%d  activation=%s",
            [k for k, v in [
                ("image",   image_encoder),
                ("tabular", tabular_encoder),
                ("text",    text_encoder),
            ] if v is not None],
            fusion_strategy, fusion_dim, problem_type, output_dim,
            type(activation).__name__,
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        tabular: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Route modalities through encoders, fuse, then predict.

        Missing-modality contract
        -------------------------
        If an encoder is registered but its corresponding input is ``None``,
        a ``torch.full((N, encoder_output_dim), 1e-7)`` dummy tensor is
        injected to preserve the fixed concatenated dimension.  This prevents
        both shape crashes and NaN propagation from pure-zero padding.

        Parameters
        ----------
        image : torch.Tensor | None
            Float tensor of shape ``(N, 3, H, W)``.
        tabular : torch.Tensor | None
            Float tensor of shape ``(N, n_features)``.
        text : List[str] | None
            List of N raw text strings.

        Returns
        -------
        torch.Tensor
            Shape ``(N, num_classes)`` — class probabilities (classification).
            Shape ``(N, 1)``           — raw scalar predictions (regression).

        Raises
        ------
        ValueError
            If all inputs are ``None`` (cannot determine batch size).
        """
        # ── Resolve batch size and reference device ───────────────────────
        # These are needed to construct correctly-shaped, correctly-placed
        # dummy tensors for absent modalities.
        batch_size: Optional[int]       = None
        ref_device: Optional[torch.device] = None

        if image is not None:
            batch_size = image.shape[0]
            ref_device = image.device
        elif tabular is not None:
            batch_size = tabular.shape[0]
            ref_device = tabular.device
        elif text is not None:
            batch_size = len(text)
            # Device from the text encoder's parameters
            if self.text_encoder is not None:
                ref_device = next(self.text_encoder.parameters()).device

        if batch_size is None:
            raise ValueError(
                "MultimodalPredictor.forward: all inputs are None. "
                "Provide at least one non-None modality input."
            )

        # Fallback device: first available encoder's parameters
        if ref_device is None:
            for enc in (self.image_encoder, self.tabular_encoder, self.text_encoder):
                if enc is not None:
                    ref_device = next(enc.parameters()).device
                    break

        features: List[torch.Tensor] = []

        # ── Image ──────────────────────────────────────────────────────────
        if self.image_encoder is not None:
            if image is not None:
                features.append(self.image_encoder(image))
            else:
                # Dummy: [N, 512] filled with 1e-7 to prevent NaN
                dummy = torch.full(
                    (batch_size, self.image_encoder.get_output_dim()),
                    1e-7,
                    dtype=torch.float32,
                    device=ref_device,
                )
                features.append(dummy)

        # ── Tabular ────────────────────────────────────────────────────────
        if self.tabular_encoder is not None:
            if tabular is not None:
                features.append(self.tabular_encoder(tabular))
            else:
                # Dummy: [N, 16] filled with 1e-7
                dummy = torch.full(
                    (batch_size, self.tabular_encoder.get_output_dim()),
                    1e-7,
                    dtype=torch.float32,
                    device=ref_device,
                )
                features.append(dummy)

        # ── Text ───────────────────────────────────────────────────────────
        if self.text_encoder is not None:
            if text is not None:
                features.append(self.text_encoder(text))
            else:
                # Dummy: [N, 768] filled with 1e-7
                dummy = torch.full(
                    (batch_size, self.text_encoder.get_output_dim()),
                    1e-7,
                    dtype=torch.float32,
                    device=ref_device,
                )
                features.append(dummy)

        # features is guaranteed non-empty here (at least one encoder exists)
        # ── Fusion → Fusion MLP → Predictor head ──────────────────────────
        fused: torch.Tensor   = self.fusion(features)
        mlp_out: torch.Tensor = self.fusion_mlp(fused)
        return self.predictor_head(mlp_out)
