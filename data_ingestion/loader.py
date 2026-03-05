"""
Lazy data loader for memory-safe multimodal dataset handling.

Tabular:  polars.scan_csv / scan_parquet  -> LazyFrame  (zero data in RAM)
          dask.dataframe                  -> DataFrame   (fallback if polars absent)
Images:   LazyImageDataset (PyTorch)      -> only paths in RAM; pixels read
          strictly inside __getitem__ via PIL
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Optional heavy-import guard: polars preferred, dask as fallback
# ---------------------------------------------------------------------------
try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

if not _POLARS_AVAILABLE:
    try:
        import dask.dataframe as dd
        _DASK_AVAILABLE = True
    except ImportError:
        _DASK_AVAILABLE = False
else:
    _DASK_AVAILABLE = False

import torch
from torch.utils.data import Dataset
from PIL import Image


# ---------------------------------------------------------------------------
# Lazy image dataset
# ---------------------------------------------------------------------------

class LazyImageDataset(Dataset):
    """
    PyTorch Dataset that stores ONLY file paths in memory.

    The actual image bytes are read from disk strictly inside ``__getitem__``.
    No pixel data is held in RAM during ``__init__`` or between ``__getitem__``
    calls.

    Args:
        image_paths: Sequence of absolute or relative paths to image files.
        transform:   Optional torchvision transform applied after PIL open.
    """

    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Any] = None,
    ) -> None:
        self._paths: List[str] = image_paths
        self._transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self._paths[idx]
        # Image bytes are read here – not before, not after
        image = Image.open(path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return {"image": image, "path": path}

    def __repr__(self) -> str:
        return f"LazyImageDataset(n_paths={len(self._paths)})"


# ---------------------------------------------------------------------------
# Internal helper: lazy tabular scan
# ---------------------------------------------------------------------------

# Union type for the lazy tabular references this module can return
LazyTabular = Union["pl.LazyFrame", "dd.DataFrame"]  # type: ignore[name-defined]


def _lazy_scan(filepath: Path) -> LazyTabular:
    """
    Return a fully lazy tabular reference – never materialises data.

    Preference order:
      1. Polars LazyFrame  (scan_parquet / scan_csv)
      2. Dask DataFrame    (read_parquet / read_csv)
    """
    if _POLARS_AVAILABLE:
        if filepath.suffix == ".parquet":
            return pl.scan_parquet(str(filepath))
        return pl.scan_csv(str(filepath))

    if _DASK_AVAILABLE:
        if filepath.suffix == ".parquet":
            return dd.read_parquet(str(filepath))  # type: ignore[union-attr]
        return dd.read_csv(str(filepath))  # type: ignore[union-attr]

    raise ImportError(
        "Neither polars nor dask is installed. "
        "Install one with: pip install polars  OR  pip install dask[dataframe]"
    )


# ---------------------------------------------------------------------------
# Public DataLoader class
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Universal lazy data loader for multimodal datasets.

    All tabular methods return a Polars ``LazyFrame`` or Dask ``DataFrame`` –
    no rows are loaded into RAM until an explicit ``.collect()`` / ``.compute()``
    call downstream.

    Images are wrapped in a ``LazyImageDataset`` (PyTorch ``Dataset``); pixels
    are read on demand inside ``__getitem__``.
    """

    # ------------------------------------------------------------------
    # Tabular lazy loaders
    # ------------------------------------------------------------------

    def load_csv(self, filepath: str) -> LazyTabular:
        """Lazily scan a CSV file.  No rows are read into RAM."""
        return _lazy_scan(Path(filepath))

    def load_parquet(self, filepath: str) -> LazyTabular:
        """Lazily scan a Parquet file.  No rows are read into RAM."""
        return _lazy_scan(Path(filepath))

    # ------------------------------------------------------------------
    # Tiny config-style loader (not lazy – these are never large)
    # ------------------------------------------------------------------

    def load_json(self, filepath: str) -> Dict[str, Any]:
        """Load a JSON file fully into memory (config/metadata use-case only)."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Image lazy loader
    # ------------------------------------------------------------------

    def load_images(
        self,
        image_paths: List[str],
        transform: Optional[Any] = None,
    ) -> LazyImageDataset:
        """
        Return a lazy PyTorch Dataset.

        Only the file *paths* are held in memory; pixels are read from disk
        strictly inside ``__getitem__`` on each access.
        """
        return LazyImageDataset(image_paths, transform=transform)

    # ------------------------------------------------------------------
    # Cache-directory auto-detect
    # ------------------------------------------------------------------

    def load_cached(
        self,
        cache_path: Path,
    ) -> Optional[Union[LazyTabular, LazyImageDataset]]:
        """
        Return a lazy reference to whatever data lives in *cache_path*.

        Detection priority:
          1. ``*.parquet`` – tabular lazy scan
          2. ``*.csv``     – tabular lazy scan
          3. ``*.jpg / *.jpeg / *.png`` (recursive) – LazyImageDataset

        Returns ``None`` if the directory is empty or contains no recognised
        file types.
        """
        if not cache_path.exists():
            return None

        # Parquet preferred
        parquet_files = sorted(cache_path.glob("*.parquet"))
        if parquet_files:
            return _lazy_scan(parquet_files[0])

        # CSV fallback
        csv_files = sorted(cache_path.glob("*.csv"))
        if csv_files:
            return _lazy_scan(csv_files[0])

        # Image directory – use itertools.islice to cap memory usage.
        # Only the first 500_000 paths are collected; for larger datasets
        # callers should use load_images() with an explicit manifest.
        import itertools
        image_iter = itertools.chain(
            cache_path.rglob("*.jpg"),
            cache_path.rglob("*.jpeg"),
            cache_path.rglob("*.png"),
        )
        image_files = list(itertools.islice(image_iter, 500_000))
        if image_files:
            return LazyImageDataset([str(p) for p in image_files])

        return None

    # ------------------------------------------------------------------
    # Pass-through merge
    # ------------------------------------------------------------------

    def merge_datasets(
        self,
        datasets: Dict[str, Union[LazyTabular, LazyImageDataset]],
    ) -> Dict[str, Union[LazyTabular, LazyImageDataset]]:
        """
        Pass-through: downstream phases handle modality-specific merging.
        Kept for API compatibility with callers that call this method.
        """
        return datasets
