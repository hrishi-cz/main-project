"""
AutoVision+ ECG/PTB-XL Smart Adapter

Handles both metadata-only PTB-XL datasets and the PTB-XL ECG image dataset
(GMC2024 synthetic ECG waveform images).

Capabilities
------------
- Detect ECG datasets by column heuristics
- Find image path columns pointing to ECG waveform PNGs/JPGs
- Resolve relative image paths against a dataset root directory
- Infer the best target/label column for classification
- Expand scp_codes dict-strings into feature columns
- Provide ECG-specific image preprocessing config (aspect ratio, normalization)
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class ECGAdapter:
    """Domain adapter for PTB-XL and ECG waveform image datasets."""

    ECG_KEYWORDS = [
        "scp_codes",
        "diagnostic",
        "superdiagnostic",
        "ecg",
        "report",
        "ecg_id",
        "filename",
        "patient_id",
        "heart_axis",
        "infarction",
        "stach",
    ]

    # Common ECG image file extensions
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    # Pattern for columns that likely hold file paths
    _PATH_PATTERN = re.compile(
        r"(filename|file_path|image_path|img_path|ecg_image|image_file|filepath|path)",
        re.IGNORECASE,
    )

    # ----------------------------------------------------------------
    # Detection
    # ----------------------------------------------------------------

    def is_ecg_dataset(self, df: pd.DataFrame) -> bool:
        """Detect PTB-XL style dataset (metadata or image variant)."""
        cols = [c.lower() for c in df.columns]
        hits = sum(any(k in c for k in self.ECG_KEYWORDS) for c in cols)
        return hits >= 1

    def is_ecg_image_dataset(self, df: pd.DataFrame, dataset_root: Optional[str] = None) -> bool:
        """
        Detect if this is an ECG *image* dataset (not just metadata).

        Returns True if the dataframe contains a column whose values look
        like file paths pointing to image files.
        """
        img_cols = self.find_image_columns(df, dataset_root)
        return len(img_cols) > 0

    # ----------------------------------------------------------------
    # Image column discovery
    # ----------------------------------------------------------------

    def find_image_columns(
        self,
        df: pd.DataFrame,
        dataset_root: Optional[str] = None,
    ) -> List[str]:
        """
        Find columns that contain paths to ECG image files.

        Detection strategy (in order):
        1. Columns whose name matches _PATH_PATTERN
        2. String columns where >30% of non-null values end with an image extension
        """
        image_cols: List[str] = []

        for col in df.columns:
            if df[col].dtype != object:
                continue

            # Strategy 1: column name heuristic
            if self._PATH_PATTERN.search(col):
                sample = df[col].dropna().head(20)
                if len(sample) > 0 and any(self._looks_like_image_path(str(v)) for v in sample):
                    image_cols.append(col)
                    continue

            # Strategy 2: value-based check – sample up to 50 rows
            sample = df[col].dropna().head(50)
            if len(sample) == 0:
                continue
            img_count = sum(1 for v in sample if self._looks_like_image_path(str(v)))
            if img_count / len(sample) > 0.3:
                image_cols.append(col)

        return image_cols

    def resolve_image_paths(
        self,
        df: pd.DataFrame,
        image_col: str,
        dataset_root: str,
    ) -> pd.Series:
        """
        Resolve relative image paths in *image_col* to absolute paths
        using *dataset_root* as the base directory.

        Handles:
        - Already-absolute paths (returned unchanged)
        - Relative paths like ``records100/00000/00001_lr.png``
        - Paths with forward/backslash mixing
        """
        root = Path(dataset_root)

        def _resolve(val: Any) -> str:
            if pd.isna(val):
                return ""
            s = str(val).replace("\\", "/").strip()
            p = Path(s)
            if p.is_absolute() and p.exists():
                return str(p)
            candidate = root / s
            if candidate.exists():
                return str(candidate)
            # Try one level up (some datasets nest CSV inside a subfolder)
            candidate2 = root.parent / s
            if candidate2.exists():
                return str(candidate2)
            return str(candidate)  # return best guess

        return df[image_col].apply(_resolve)

    # ----------------------------------------------------------------
    # Target inference
    # ----------------------------------------------------------------

    def infer_ecg_target(self, df: pd.DataFrame) -> str:
        """Find the best ECG label column by priority."""
        priority = [
            "diagnostic_superclass",
            "diagnostic_class",
            "scp_codes",
            "superdiagnostic",
            "label",
            "class",
        ]
        lower_map = {c.lower(): c for c in df.columns}
        for p in priority:
            if p in lower_map:
                return lower_map[p]
        return "Unknown"

    # ----------------------------------------------------------------
    # SCP codes expansion
    # ----------------------------------------------------------------

    def expand_scp_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand scp_codes dict-string column into numeric features.

        Creates:
        - ``scp_codes_len``: number of diagnoses per record
        - ``scp_primary_code``: the dominant diagnostic code (highest score)
        """
        if "scp_codes" not in df.columns:
            return df

        try:
            parsed = df["scp_codes"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("{") else {}
            )
            df["scp_codes_len"] = parsed.apply(len)
            df["scp_primary_code"] = parsed.apply(
                lambda d: max(d, key=d.get) if d else "UNKNOWN"
            )
        except Exception as exc:
            logger.warning("ECGAdapter: could not expand scp_codes: %s", exc)

        return df

    # ----------------------------------------------------------------
    # ECG-specific image preprocessing config
    # ----------------------------------------------------------------

    def get_image_config(self) -> Dict[str, Any]:
        """
        Return ECG-specific image preprocessing configuration.

        ECG waveform images are typically:
        - Wider than tall (12-lead ECG strips)
        - Grayscale or limited color
        - Need different normalization than natural images
        """
        return {
            "target_size": (224, 448),  # height x width – preserve landscape aspect
            "grayscale": False,         # keep RGB (grid lines are colored in some datasets)
            "normalize": {
                # ECG image-specific normalization (closer to 0.5 mean)
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "augmentations": [
                "random_horizontal_flip",   # ECG is symmetric in some orientations
                "random_brightness",        # paper/screen background variation
                "random_contrast",          # ink/line thickness variation
            ],
        }

    # ----------------------------------------------------------------
    # Dataset summary
    # ----------------------------------------------------------------

    def summarize(self, df: pd.DataFrame, dataset_root: Optional[str] = None) -> Dict[str, Any]:
        """
        Produce a structured summary of the ECG dataset for logging/UI.
        """
        image_cols = self.find_image_columns(df, dataset_root)
        target_col = self.infer_ecg_target(df)
        has_images = len(image_cols) > 0

        # Count how many image files actually exist on disk
        existing_images = 0
        total_images = 0
        if has_images and dataset_root:
            sample = df[image_cols[0]].dropna().head(100)
            total_images = len(df)
            root = Path(dataset_root)
            for v in sample:
                p = root / str(v).replace("\\", "/")
                if p.exists():
                    existing_images += 1

        summary: Dict[str, Any] = {
            "is_ecg": True,
            "has_images": has_images,
            "image_columns": image_cols,
            "target_column": target_col,
            "total_records": len(df),
            "columns": list(df.columns),
        }
        if has_images:
            summary["image_files_sampled"] = len(sample) if dataset_root else 0
            summary["image_files_found"] = existing_images
            summary["estimated_total_images"] = total_images

        return summary

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _looks_like_image_path(value: str) -> bool:
        """Check if a string value looks like a path to an image file."""
        v = value.strip().lower().replace("\\", "/")
        if not v:
            return False
        # Must have a path separator or end with image extension
        return any(v.endswith(ext) for ext in ECGAdapter._IMAGE_EXTS)
