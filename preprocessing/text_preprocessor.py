"""Text preprocessing – HuggingFace BERT tokeniser (PyTorch DataLoader-safe)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    BERT-compatible text preprocessor.

    Tokenises raw strings with ``bert-base-uncased`` via HuggingFace
    ``AutoTokenizer``.  Output tensors are squeezed from ``[1, 128]`` to
    ``[128]`` so that ``DataLoader`` batching produces ``[B, 128]`` rather
    than ``[B, 1, 128]`` (which would crash Phase 5 model forward passes).

    Usage
    -----
    >>> tp = TextPreprocessor()
    >>> out = tp("Diagnosis shows severe inflammation.")
    >>> out["input_ids"].shape
    torch.Size([128])
    """

    _PRETRAINED: str = "bert-base-uncased"
    _MAX_LENGTH: int = 128

    def __init__(self) -> None:
        self._tokenizer: Optional[Any] = None  # lazy-loaded on first call

    # ------------------------------------------------------------------
    # Lazy tokeniser property
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        """Load the BERT tokeniser once; re-use thereafter."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self._PRETRAINED)
                logger.info("TextPreprocessor: loaded tokeniser '%s'", self._PRETRAINED)
            except Exception as exc:
                raise RuntimeError(
                    f"TextPreprocessor: could not load '{self._PRETRAINED}'. "
                    "Install transformers: pip install transformers"
                ) from exc
        return self._tokenizer

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenise *text* and return a dict of fixed-length tensors.

        Returns
        -------
        dict with keys:
          ``input_ids``      : ``torch.LongTensor`` of shape ``[128]``
          ``attention_mask`` : ``torch.LongTensor`` of shape ``[128]``

        The ``.squeeze(0)`` call converts the tokeniser's ``[1, 128]``
        output to ``[128]`` so DataLoader can stack a batch to ``[B, 128]``.
        """
        # Sanitize NaN/None inputs — tokenizing "nan"/"None" as words produces
        # misleading embeddings.  Replace with empty string instead.
        if text is None:
            text = ""
        else:
            text = str(text)
            if text.lower() in ("nan", "none", "null", "<na>"):
                text = ""

        encoding = self.tokenizer(
            text,
            max_length=self._MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),           # [128]
            "attention_mask": encoding["attention_mask"].squeeze(0), # [128]
        }

    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        """Delegate to :meth:`preprocess` – makes instances callable as transforms."""
        return self.preprocess(text)

    # ------------------------------------------------------------------
    # Config helper (used by /preprocess API endpoint)
    # ------------------------------------------------------------------

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model": self._PRETRAINED,
            "max_length": self._MAX_LENGTH,
            "padding": "max_length",
            "truncation": True,
            "output_keys": ["input_ids", "attention_mask"],
            "output_shape": f"[{self._MAX_LENGTH}] per key",
        }
