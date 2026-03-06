"""Tabular preprocessing – ColumnTransformer pipeline (PyTorch-compatible)."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic thresholds for automatic column filtering
# ---------------------------------------------------------------------------
_MAX_OHE_CARDINALITY: int = 50          # one-hot encode only if ≤ this many unique values
_NEAR_UNIQUE_RATIO: float = 0.5         # drop if unique_ratio > this (likely IDs)
_PATH_PATTERN = re.compile(              # detect file paths / URLs
    r"[/\\]|\.(?:csv|json|dat|hea|png|jpg|wav|mp3|parquet|zip)", re.IGNORECASE
)
_ID_NAME_PATTERN = re.compile(           # detect ID-like column names
    r"(?:^|_)(?:id|idx|index|key|serial|pk|fk)(?:$|_)",
    re.IGNORECASE,
)
_NEAR_UNIQUE_RATIO_STRICT: float = 0.9  # drop ANY integer col above this
_ID_UNIQUE_RATIO: float = 0.1           # drop ID-named cols above this


class TabularPreprocessor:
    """
    Scikit-learn ColumnTransformer pipeline that produces ``np.float32``
    arrays ready for ``torch.tensor()``.

    Pipeline
    --------
    Numeric columns  : SimpleImputer(median) → StandardScaler
    Categorical cols : SimpleImputer(most_frequent) → OneHotEncoder(sparse=False)

    Automatic filtering (universal pipeline safety):
    - Columns with near-unique values (>50% unique) are dropped (IDs, paths, timestamps).
    - Categorical columns with >50 unique values are dropped (prevents OHE explosion).
    - File-path and URL columns are detected and dropped.

    ``sparse_output=False`` is mandatory: ``torch.tensor()`` cannot consume
    scipy sparse matrices and will raise a ``TypeError``.

    Usage
    -----
    >>> tp = TabularPreprocessor()
    >>> arr = tp.fit_transform(train_df)      # np.float32, shape (N, D)
    >>> arr_test = tp.transform(test_df)      # same D
    >>> dim = tp.get_output_dim()
    """

    def __init__(self) -> None:
        self._transformer: Optional[ColumnTransformer] = None
        self._feature_names_in: List[str] = []
        self._dropped_cols: List[str] = []

    # ------------------------------------------------------------------
    # Column filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _is_path_column(series: pd.Series) -> bool:
        """Return True if most non-null values look like file paths or URLs."""
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        return _PATH_PATTERN.search(sample.iloc[0]) is not None and (
            sample.str.contains(r"[/\\]", na=False).mean() > 0.5
        )

    @staticmethod
    def _is_datetime_like(series: pd.Series) -> bool:
        """Return True if a string/object column looks like dates or timestamps."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        sample = series.dropna().astype(str).head(30)
        if len(sample) == 0:
            return False
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            return parsed.notna().mean() > 0.7
        except Exception:
            return False

    def _filter_categorical(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
    ) -> List[str]:
        """
        Remove categorical columns that would cause OHE explosion.

        Drops:
        - Near-unique columns (unique ratio > 50%) – IDs, hashes, timestamps
        - High-cardinality columns (> _MAX_OHE_CARDINALITY unique values)
        - File-path / URL columns
        """
        kept: List[str] = []
        n_rows = len(df)

        for col in categorical_cols:
            n_unique = df[col].nunique(dropna=True)
            unique_ratio = n_unique / max(n_rows, 1)

            # Check 1: near-unique → likely an ID or hash
            if unique_ratio > _NEAR_UNIQUE_RATIO:
                logger.info(
                    "  DROP '%s': near-unique (%.0f%% unique, %d values) – likely ID/hash",
                    col, unique_ratio * 100, n_unique,
                )
                self._dropped_cols.append(col)
                continue

            # Check 2: file-path column
            if self._is_path_column(df[col]):
                logger.info("  DROP '%s': detected as file-path column", col)
                self._dropped_cols.append(col)
                continue

            # Check 3: datetime-like string column
            if self._is_datetime_like(df[col]):
                logger.info("  DROP '%s': detected as datetime string column", col)
                self._dropped_cols.append(col)
                continue

            # Check 4: too many unique values for OHE
            if n_unique > _MAX_OHE_CARDINALITY:
                logger.info(
                    "  DROP '%s': cardinality %d exceeds OHE limit (%d)",
                    col, n_unique, _MAX_OHE_CARDINALITY,
                )
                self._dropped_cols.append(col)
                continue

            kept.append(col)

        return kept

    def _filter_numeric(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
    ) -> List[str]:
        """
        Remove numeric columns that carry no signal.

        Drops:
        - Constant columns (zero variance)
        - Near-unique integer columns with ID-like names (e.g. patient_id, ecg_id)
        - Very high uniqueness integers (>90% unique) regardless of name
        - Datetime-typed numeric columns (e.g. Unix timestamps stored as int64)
        """
        kept: List[str] = []
        n_rows = len(df)

        for col in numeric_cols:
            n_unique = df[col].nunique(dropna=True)

            # Constant column
            if n_unique <= 1:
                logger.info("  DROP '%s': constant (single value)", col)
                self._dropped_cols.append(col)
                continue

            # Integer column checks
            if pd.api.types.is_integer_dtype(df[col]):
                unique_ratio = n_unique / max(n_rows, 1)

                # Check 1: Column name matches ID pattern + any meaningful uniqueness
                if _ID_NAME_PATTERN.search(col) and unique_ratio > _ID_UNIQUE_RATIO:
                    logger.info(
                        "  DROP '%s': ID-like name + %.0f%% unique values",
                        col, unique_ratio * 100,
                    )
                    self._dropped_cols.append(col)
                    continue

                # Check 2: Very high uniqueness (>90%) — almost certainly auto-increment
                if unique_ratio > _NEAR_UNIQUE_RATIO_STRICT:
                    logger.info(
                        "  DROP '%s': near-unique integer (%.0f%% unique) "
                        "– likely auto-increment ID",
                        col, unique_ratio * 100,
                    )
                    self._dropped_cols.append(col)
                    continue

            # Datetime-typed numeric column (e.g. datetime64 parsed as nanoseconds)
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                logger.info("  DROP '%s': datetime numeric column", col)
                self._dropped_cols.append(col)
                continue

            kept.append(col)

        return kept

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """
        Fit the ColumnTransformer on *df*.

        Numeric and categorical columns are detected automatically.
        Useless columns (IDs, paths, datetimes, high-cardinality) are dropped.
        """
        self._dropped_cols = []

        # Auto-detect and drop datetime64 columns before numeric/categorical split
        datetime_cols: List[str] = df.select_dtypes(
            include=["datetime64", "datetimetz"]
        ).columns.tolist()
        if datetime_cols:
            logger.info(
                "  DROP %d datetime columns: %s",
                len(datetime_cols), datetime_cols,
            )
            self._dropped_cols.extend(datetime_cols)
            df = df.drop(columns=datetime_cols)

        numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols: List[str] = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Smart filtering
        numeric_cols = self._filter_numeric(df, numeric_cols)
        categorical_cols = self._filter_categorical(df, categorical_cols)

        self._feature_names_in = numeric_cols + categorical_cols

        if self._dropped_cols:
            logger.info(
                "TabularPreprocessor: dropped %d useless columns: %s",
                len(self._dropped_cols),
                self._dropped_cols,
            )

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = []
        if numeric_cols:
            transformers.append(("numeric", numeric_pipeline, numeric_cols))
        if categorical_cols:
            transformers.append(("categorical", categorical_pipeline, categorical_cols))

        if not transformers:
            raise ValueError(
                "TabularPreprocessor.fit: DataFrame has no usable numeric or categorical columns "
                f"(dropped {len(self._dropped_cols)} column(s) as IDs/paths/high-cardinality)."
            )

        self._transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )
        self._transformer.fit(df)
        logger.info(
            "TabularPreprocessor fitted: %d numeric, %d categorical columns → output dim %d",
            len(numeric_cols),
            len(categorical_cols),
            self.get_output_dim(),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted transformer.

        Returns
        -------
        np.ndarray of dtype ``float32``, shape ``(N, output_dim)``.
        """
        if self._transformer is None:
            raise RuntimeError("TabularPreprocessor must be fitted before transform().")
        # Align columns to match the fitted feature set
        if self._feature_names_in:
            missing = [c for c in self._feature_names_in if c not in df.columns]
            if missing:
                df = df.copy()
                for col in missing:
                    df[col] = 0.0
                logger.warning(
                    "TabularPreprocessor.transform: zero-filled %d missing columns: %s",
                    len(missing), missing,
                )
            df = df[self._feature_names_in]
        result = self._transformer.transform(df)
        return result.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_output_dim(self) -> int:
        """Return the number of output features after transformation."""
        if self._transformer is None:
            return 0
        try:
            return sum(
                len(t.get_feature_names_out())
                for _, t, _ in self._transformer.transformers_
                if hasattr(t, "get_feature_names_out")
                   and not isinstance(t, str)
            )
        except Exception:
            # Fallback: derive output dim from a single-row transform.
            # n_features_in_ is the INPUT count (pre-OHE), not output.
            try:
                dummy = self._transformer.transform(
                    pd.DataFrame(
                        np.zeros((1, self._transformer.n_features_in_)),
                        columns=self._transformer.feature_names_in_,
                    )
                )
                return dummy.shape[1]
            except Exception:
                return getattr(self._transformer, "n_features_in_", 0)

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "numeric_pipeline": ["SimpleImputer(median)", "StandardScaler"],
            "categorical_pipeline": ["SimpleImputer(most_frequent)", "OneHotEncoder(sparse=False)"],
            "output_dtype": "float32",
            "output_shape": "(N, output_dim)",
        }
