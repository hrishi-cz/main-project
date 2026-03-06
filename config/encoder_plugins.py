"""
config/encoder_plugins.py

User-configurable encoder plugin registration.

Add your custom encoders here.  They will be included in the JIT
encoder selection pool alongside the built-in encoders (ConvNeXt-Tiny,
ResNet-50, MobileNetV3, DeBERTa, BERT, MiniLM, GRN, MLP).

This file is imported at startup.  All registrations happen at
module-import time so the encoders are available when the JIT
selector runs.
"""

# from automl.jit_encoder_selector import (
#     register_vision_encoder,
#     register_text_encoder,
#     register_tabular_encoder,
# )

# ── Example: Register a custom vision encoder ──────────────────────────
#
# import torch
# import torch.nn as nn
# from automl.jit_encoder_selector import _freeze_and_eval
#
# def _make_efficientnet_b0() -> nn.Module:
#     from torchvision import models
#     backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
#     out_features = backbone.classifier[1].in_features  # 1280
#     backbone.classifier = nn.Identity()
#     projection = nn.Sequential(nn.Linear(out_features, 512), nn.ReLU())
#     # Wrap in a module with get_output_dim()
#     class _Wrapper(nn.Module):
#         def __init__(self, bb, proj):
#             super().__init__()
#             self.backbone = bb
#             self.projection = proj
#             self._output_dim = 512
#         def forward(self, x):
#             return self.projection(self.backbone(x))
#         def get_output_dim(self):
#             return self._output_dim
#     encoder = _Wrapper(backbone, projection)
#     return _freeze_and_eval(encoder)
#
# register_vision_encoder(
#     name="EfficientNet-B0",
#     factory=_make_efficientnet_b0,
#     output_dim=512,
#     capacity=5_300_000,
#     dummy_input_fn=lambda bs, dev: torch.randn(bs, 3, 224, 224, device=dev),
# )

# ── Example: Register a custom tabular encoder ─────────────────────────
#
# import torch.nn as nn
#
# class MyTabularEncoder(nn.Module):
#     def __init__(self, input_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16),
#         )
#         self._output_dim = 16
#     def forward(self, x):
#         return self.net(x)
#     def get_output_dim(self):
#         return self._output_dim
#
# register_tabular_encoder(
#     name="MyTabular",
#     encoder_class=MyTabularEncoder,
#     output_dim=16,
#     capacity=8_000,
# )
