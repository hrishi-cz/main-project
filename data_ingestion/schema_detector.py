from __future__ import annotations

import logging
import re as _re_import
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


from data_ingestion.schema import GlobalSchema, IndividualSchema
from data_ingestion.integrator import Integrator
from data_ingestion.data_bridge import materialize_sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TargetScore dataclass (used throughout for candidate scoring)
# ---------------------------------------------------------------------------
@dataclass
class TargetScore:
    column: str
    final_score: float
    nan_ratio: float = 0.0
    valid: bool = True
    reason: str = "Valid"
    quality: float = 0.0
    semantic_score: float = 0.0
    semantics: Dict[str, Any] = field(default_factory=dict)
    explanation: List[str] = field(default_factory=list)
    semantic_role: str = ""
    interaction_score: float = 0.0
    uncertainty: float = 0.0
    # --- Previously missing fields ---
    keyword_score: float = 0.0
    uniqueness_score: float = 0.0
    regression_score: float = 0.0
    json_score: float = 0.0
    predictability_score: float = 0.0
    complementarity_score: float = 0.0
    cross_dataset_score: float = 0.0
    degeneracy_penalty: float = 0.0


# ===========================================================================
# Main Engine
# ===========================================================================

class COGMASchemaDetector:
    """
    COGMA-ready schema detector: 6-stage intelligence pipeline.
    All detection/validation flows through Integrator. No heuristics.
    """

    TARGET_KEYWORDS = ["target", "label", "class", "output", "diagnosis", "sentiment", "code"]
    TARGET_SUFFIX_KEYWORDS = ["id", "val", "score"]
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    _BINARY_ATTRIBUTE_VALUES = {"yes", "no", "true", "false", "0", "1", "m", "f"}

    def __init__(self, fix4_engine=None):
        self.fix4_engine = fix4_engine
        self.last_target_candidates: List[TargetScore] = []

    def _build_global_schema(self, per_dataset_results: List[Dict[str, Any]]) -> GlobalSchema:
        """Aggregate per-dataset results into a GlobalSchema."""
        global_modalities = sorted({m for s in per_dataset_results for m in s.get("modalities", [])})
        
        if per_dataset_results:
            primary_target = max(per_dataset_results, key=lambda s: s.get("confidence", 0.0)).get("target_column", "Unknown")
            detection_confidence = float(np.mean([s.get("confidence", 0.0) for s in per_dataset_results]))
        else:
            primary_target = "Unknown"
            detection_confidence = 0.0
            
        fusion_ready = len(global_modalities) > 1

        global_schema = GlobalSchema(
            global_problem_type="classification_multiclass",  # Could be refined
            global_modalities=global_modalities,
            primary_target=primary_target,
            fusion_ready=fusion_ready,
            detection_confidence=round(detection_confidence, 3),
            per_dataset=per_dataset_results,
        )
        return global_schema

    def _select_primary_target(self, per_dataset_results: List[Dict[str, Any]]) -> str:
        """
        Select the primary target column across datasets, boosting targets that appear in multiple datasets.
        """
        target_scores = {}  # Initialize target scores
        target_appearances = {}

        for s in per_dataset_results:
            t = s.get("target_column")
            score = s.get("confidence", 0)
            if t and t != "Unknown":
                target_scores[t] = target_scores.get(t, 0) + score
                target_appearances[t] = target_appearances.get(t, 0) + 1

        # Boost score for targets appearing in multiple datasets (cross-dataset bonus)
        for target in target_scores:
            count = target_appearances[target]
            # Linear boost: appearing in N datasets → +0.1 * (N-1)
            cross_dataset_bonus = 0.1 * max(0, count - 1)
            target_scores[target] += cross_dataset_bonus

        if target_scores:
            return max(target_scores, key=target_scores.get)
        return "Unknown"

    def _compute_modality_importance(self, per_dataset_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute the importance of each modality based on confidence scores across datasets.
        """
        scores = {"tabular": 0.0, "text": 0.0, "image": 0.0, "timeseries": 0.0}
        for ds in per_dataset_results:
            conf = float(ds.get("confidence", 0.0))
            for m in ds.get("modalities", []):
                if m in scores:
                    scores[m] += conf
        total = sum(scores.values()) or 1.0
        return {k: round(v / total, 3) for k, v in scores.items()}

    # -----------------------------------------------------------------------
    # PATCH C1: Image target scoring
    # -----------------------------------------------------------------------

    def _score_image_target_candidates(
        self, sample_df: pd.DataFrame, image_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Score candidate target columns for image datasets. Returns ranked list of candidate dicts with final_score.
        """
        candidates = []
        non_image_cols = [c for c in sample_df.columns if c not in image_cols]

        for col in non_image_cols:
            series = sample_df[col].dropna()
            if len(series) == 0:
                continue

            n_unique   = series.nunique()
            nan_ratio  = 1.0 - len(series) / max(len(sample_df), 1)
            col_lower  = col.lower()

            # 1. Cardinality score → prefer [2, 100]
            if n_unique < 2:
                cardinality_score = 0.0
            elif n_unique <= 10:
                cardinality_score = 1.0
            elif n_unique <= 50:
                cardinality_score = 0.7
            elif n_unique <= 100:
                cardinality_score = 0.4
            else:
                cardinality_score = 0.1  # likely a path/id column

            # 2. Class balance score (Gini-based evenness)
            try:
                val_counts = series.value_counts(normalize=True)
                balance = 1.0 - float(val_counts.std())  # high std → imbalanced
                balance_score = max(0.0, min(1.0, balance))
            except Exception:
                balance_score = 0.5

            # 3. Semantic keyword match
            sem_hits = [kw for kw in self.TARGET_KEYWORDS if kw in col_lower]
            semantic_score = min(1.0, len(sem_hits) * 0.4)

            # 4. Bbox / JSON detection penalty
            try:
                sample_str = str(series.iloc[0])
                is_bbox = sample_str.startswith("{") or "xmin" in col_lower or "bbox" in col_lower
                bbox_penalty = 0.3 if is_bbox else 0.0
            except Exception:
                bbox_penalty = 0.0

            final = (
                0.35 * cardinality_score
                + 0.25 * balance_score
                + 0.25 * semantic_score
                - 0.15 * nan_ratio
                - bbox_penalty
            )

            candidates.append({
                "column":           col,
                "cardinality_score": round(cardinality_score, 3),
                "balance_score":    round(balance_score, 3),
                "semantic_score":   round(semantic_score, 3),
                "nan_ratio":        round(nan_ratio, 3),
                "final_score":      round(max(0.0, final), 3),
                "n_unique":         int(n_unique),
            })

        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)

    # -----------------------------------------------------------------------
    # PATCH C2: Text target scoring
    # -----------------------------------------------------------------------

    def _score_text_target_candidates(
        self, sample_df: pd.DataFrame, text_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Score candidate target columns for text/NLP datasets. Returns ranked list of candidate dicts with type + final_score.
        """
        import re

        _IOB_PATTERN  = re.compile(r"^(B|I|O)-[A-Z]{1,10}$")
        _SEQ2SEQ_TOKENS = {"</s>", "[SEP]", "<eos>", "[CLS]", "<pad>"}

        candidates = []
        non_text_content_cols = [c for c in sample_df.columns if c not in text_cols]

        for col in non_text_content_cols:
            series   = sample_df[col].dropna().astype(str)
            if len(series) == 0:
                continue

            col_lower = col.lower()
            n_unique  = series.nunique()
            nan_ratio = 1.0 - series.count() / max(len(sample_df), 1)
            sample_vals = series.head(50).tolist()

            target_type    = "classification"
            type_conf      = 0.0
            semantic_score = min(1.0, sum(kw in col_lower for kw in self.TARGET_KEYWORDS) * 0.35)

            # --- Classification detection ---
            if 2 <= n_unique <= 50:
                target_type = "text_classification"
                if n_unique <= 5:
                    type_conf = 0.9
                elif n_unique <= 20:
                    type_conf = 0.7
                else:
                    type_conf = 0.5

            # --- NER / IOB detection (overrides classification) ---
            iob_matches = sum(
                1 for v in sample_vals
                if isinstance(v, str) and any(_IOB_PATTERN.match(tok) for tok in v.split())
            )
            if iob_matches / max(len(sample_vals), 1) > 0.3:
                target_type = "ner_sequence"
                type_conf   = min(1.0, iob_matches / len(sample_vals) * 1.5)

            # --- Seq2seq detection ---
            avg_len = float(series.str.len().mean())
            special_tok_hits = sum(1 for v in sample_vals if any(t in v for t in _SEQ2SEQ_TOKENS))
            if avg_len > 30 and (special_tok_hits > 0 or "output" in col_lower or "response" in col_lower):
                target_type = "seq2seq"
                type_conf   = 0.65 + 0.1 * min(1.0, special_tok_hits / max(len(sample_vals), 1))

            final = (
                0.40 * type_conf
                + 0.30 * semantic_score
                - 0.15 * nan_ratio
                - (0.20 if target_type == "seq2seq" and n_unique > 100 else 0.0)  # penalise unique seq2seq
            )

            candidates.append({
                "column":      col,
                "target_type": target_type,
                "type_conf":   round(type_conf, 3),
                "semantic_score": round(semantic_score, 3),
                "n_unique":    int(n_unique),
                "avg_len":     round(avg_len, 1),
                "nan_ratio":   round(nan_ratio, 3),
                "final_score": round(max(0.0, final), 3),
            })

        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)

    def _infer_problem_from_override(
        self, col: str, datasets: Dict[str, Any]
    ) -> str:
        """
        Re-infer problem type from an override column's unique values. Materialises only the single target column (memory-safe).
        """
        try:
            import polars as pl
            for lazy_ref in datasets.values():
                if isinstance(lazy_ref, pl.LazyFrame):
                    schema_names = lazy_ref.collect_schema().names()
                    if col not in schema_names:
                        continue
                    series = lazy_ref.select(col).head(5000).collect()[col]
                    n_unique = series.n_unique()
                    if series.dtype in (pl.Utf8, pl.String, pl.Categorical):
                        return "classification_multiclass" if n_unique > 2 else "classification_binary"
                    if n_unique <= 2:
                        return "classification_binary"
                    if n_unique <= 20:
                        return "classification_multiclass"
                    return "regression"
        except Exception:
            pass
        return "classification_multiclass"  # safe default for unknown

    # -----------------------------------------------------------------------
    # Tier-1: single-dataset inspector
    # -----------------------------------------------------------------------

    def _detect_single(
        self,
        dataset_id: str,
        lazy_data: Any,
        target_override: Optional[str] = None,
    ) -> IndividualSchema:
        """
        Tier-1: inspect one lazy dataset. Materialises at most 500 rows for
        heuristic computation. For PyTorch Datasets (image-only) it returns an
        image-modality schema directly without any column analysis.

        Args:
            target_override: When set, bypass auto target detection and mark
                             this column as the target directly.
        """
        sample_df: Optional[pd.DataFrame] = self._materialise_sample(
            lazy_data, n=500
        )

        if sample_df is None:
            # PyTorch Dataset or unrecognised type → treat as image dataset
            from data_ingestion.loader import detect_image_structure
            from pathlib import Path

            cache_path = Path("./data/dataset_cache") / dataset_id
            structure = detect_image_structure(cache_path)

            if structure["type"] == "classification":
                target_col = "__image_label__"
                prob_type = "classification_multiclass"
                conf = 0.9
                reasoning = {"reason": "Detected class folders", "selected": {"column": "__image_label__", "final_score": 0.9}}
                candidates = [{"column": "__image_label__", "final_score": 0.9, "reason": "Detected class folders"}]
            else:
                target_col = "Unknown"
                prob_type = "unsupervised"
                conf = 0.5
                reasoning = {"reason": "No label structures detected"}
                candidates = [
                    {"column": "__image_label__", "final_score": 0.0, "reason": "No folder patterns"},
                    {"column": "__filename_pattern__", "final_score": 0.0, "reason": "No matched pattern"},
                    {"column": "__unsupervised__", "final_score": 0.5, "reason": "Default fallback"}
                ]

            return IndividualSchema(
                dataset_id=dataset_id,
                detected_columns={
                    "image": ["__image_path__"],
                    "text": [],
                    "tabular": [],
                    "timeseries": [],
                },
                target_column=target_col,
                problem_type=prob_type,
                modalities=["image"],
                confidence=conf,
                reasoning=reasoning,
                candidates=candidates,
            )

        return self._inspect_dataframe(dataset_id, sample_df, target_override=target_override)

    # -----------------------------------------------------------------------
    # Advanced Semantic Target Heuristics (Blueprint)
    # -----------------------------------------------------------------------

    def _safe_div(self, a: float, b: float) -> float:
        return float(a) / float(b) if b else 0.0

    def _nan_ratio(self, s: pd.Series) -> float:
        return float(s.isna().mean()) if len(s) else 1.0

    def _unique_ratio(self, s: pd.Series) -> float:
        return self._safe_div(s.nunique(dropna=True), max(len(s), 1))

    def _avg_len(self, s: pd.Series) -> float:
        try:
            sample = s.dropna().astype(str).head(50)
            return float(sample.str.len().mean()) if len(sample) else 0.0
        except Exception:
            return 0.0

    def _json_ratio(self, s: pd.Series) -> float:
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            return 0.0
        return float(sample.str.contains(r"^\s*\{.*\}\s*$", na=False).mean())

    def _list_ratio(self, s: pd.Series) -> float:
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            return 0.0
        return float(sample.str.contains(r"^\s*\[.*\]\s*$", na=False).mean())

    def _looks_like_path(self, s: pd.Series) -> float:
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            return 0.0
        return float(sample.str.contains(r"[/\\]|\.(?:png|jpg|jpeg|bmp|tif|tiff|csv|json|parquet)$", case=False, regex=True).mean())

    def _is_image_path_series(self, s: pd.Series) -> bool:
        return self._looks_like_path(s) > 0.5 and self._avg_len(s) < 250

    def _is_text_series(self, s: pd.Series) -> bool:
        if s.dtype != "object":
            return False
        return self._avg_len(s) > 30 and self._unique_ratio(s) > 0.1

    def _is_structured_label(self, s: pd.Series) -> bool:
        if s.dtype != "object":
            return False
        return (self._json_ratio(s) > 0.2) or (self._list_ratio(s) > 0.2)

    def _target_quality_score(self, s: pd.Series, name: str) -> float:
        unique_ratio = self._unique_ratio(s)
        nan_ratio = self._nan_ratio(s)
        name_l = name.lower()
        score = 0.0
        if nan_ratio < 0.5:
            score += 0.20
        if 2 <= s.nunique(dropna=True) <= 50:
            score += 0.25
        if self._is_structured_label(s):
            score += 0.35
        if s.dtype == "object" and self._avg_len(s) < 50:
            score += 0.10
        if unique_ratio > 0.98 and s.dtype != "object":
            score -= 0.35
        if any(k in name_l for k in ["target", "label", "class", "output", "code", "diagnosis", "result"]):
            score += 0.10
        return float(max(0.0, min(1.0, score)))

    def _infer_image_schema(self, dataset_id: str, lazy_data: Any) -> IndividualSchema:
        targets = getattr(lazy_data, "targets", None) or getattr(lazy_data, "labels", None)
        semantic = self._image_label_quality(lazy_data)

        detected = {"image": ["__image__"], "text": [], "tabular": [], "timeseries": []}
        reasoning = {"mode": "image", "reason": [], "notes": []}

        if semantic == 1.0 and targets is not None:
            n_classes = len(getattr(lazy_data, "classes", []))
            problem_type = "classification_binary" if n_classes == 2 else "classification_multiclass"
            reasoning["reason"].append("Detected class metadata in image dataset")
            reasoning["notes"].append(f"{n_classes} classes found")

            return IndividualSchema(
                dataset_id=dataset_id,
                detected_columns=detected,
                target_column="__image_label__",
                problem_type=problem_type,
                modalities=["image"],
                confidence=0.90,
                target_profile={"semantic_score": 1.0, "predictability_score": 0.8, "quality_score": 1.0},
                reasoning=reasoning,
                candidates=[{"column": "__image_label__", "final_score": 1.0, "reason": "Image class metadata detected", "valid": True}],
                rejected_candidates=[],
                preprocessing_hints={"image": {"mode": "supervised", "resize": [224, 224], "augment": True, "label_source": "metadata"}}
            )

        reasoning["reason"].append("No explicit image labels found")
        reasoning["notes"].append("Fallback to self-supervised / representation learning")

        return IndividualSchema(
            dataset_id=dataset_id,
            detected_columns=detected,
            target_column="Unknown",
            problem_type="unsupervised",
            modalities=["image"],
            confidence=0.50,
            target_profile={"semantic_score": 0.0, "predictability_score": 0.0, "quality_score": 0.0},
            reasoning=reasoning,
            candidates=[],
            rejected_candidates=[],
            preprocessing_hints={"image": {"mode": "self_supervised", "resize": [224, 224], "augment": True, "label_source": None}}
        )

    def _build_preprocessing_hints(self, modalities: List[str], target_col: str, problem_type: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = candidates[0] if candidates else {}
        target_type = "semantic" if best.get("semantic_score", 0) > 0.4 else "predictive"
        hints = {
            "target_type": target_type,
            "feature_selection": "minimal" if target_type == "semantic" else "strict",
            "tabular": {
                "use_mi_shap": True,
                "top_k_ratio": 0.8 if target_type == "semantic" else 0.4,
                "keep_interactions": True
            },
            "text": {
                "mode": "structured_label" if best.get("list_ratio", 0) > 0.2 or best.get("json_ratio", 0) > 0.2 else "free_text",
                "max_length": 256 if target_type == "semantic" else 128
            },
            "image": {
                "mode": "supervised" if problem_type.startswith("classification") else "self_supervised",
                "resize": [256, 256] if target_type == "semantic" else [224, 224],
                "augment": True
            },
            "multimodal": {
                "fusion_ready": len([m for m in modalities if m in ["tabular", "text", "image"]]) > 1,
                "weights": {
                    "tabular": 0.5 if "tabular" in modalities else 0.0,
                    "text": 0.4 if "text" in modalities else 0.0,
                    "image": 0.1 if "image" in modalities else 0.0,
                }
            }
        }
        return hints

    # -----------------------------------------------------------------------
    # Sample materialisation – lazy → pandas (at most n rows)
    # -----------------------------------------------------------------------

    @staticmethod
    def _materialise_sample(
        lazy_data: Any,
        n: int = 500,
    ) -> Optional[pd.DataFrame]:
        """
        Materialise at most *n* rows from a lazy reference into a pandas
        DataFrame.  Returns None for PyTorch Datasets (image-only sources).
        """
        # --- Polars LazyFrame ---
        try:
            import polars as pl  # noqa: F401 – guarded import
            if isinstance(lazy_data, pl.LazyFrame):
                return lazy_data.head(n).collect().to_pandas()
        except ImportError:
            pass

        # --- Dask DataFrame ---
        try:
            import dask.dataframe as dd  # noqa: F401
            if isinstance(lazy_data, dd.DataFrame):
                return lazy_data.head(n, compute=True)
        except ImportError:
            pass

        # --- Plain pandas DataFrame (legacy / already materialised) ---
        if isinstance(lazy_data, pd.DataFrame):
            return lazy_data.head(n)

        # --- PyTorch Dataset (image collections) ---
        try:
            from torch.utils.data import Dataset  # noqa: F401
            if isinstance(lazy_data, Dataset):
                return None  # handled upstream as image-only schema
        except ImportError:
            pass

        # Unknown type – attempt pandas coercion as last resort
        try:
            return pd.DataFrame(lazy_data).head(n)
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Core column-level analysis (runs on the pandas sample)
    # -----------------------------------------------------------------------

    def _inspect_dataframe(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        target_override: Optional[str] = None,
    ) -> IndividualSchema:
        detected: Dict[str, List[str]] = {"image": [], "text": [], "tabular": [], "timeseries": []}
        candidates: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        semantic_summary = {}
        interaction_summary = {}
        uncertainty_summary = {}

        for col in df.columns:
            s = df[col]
            if self._is_image(s):
                detected["image"].append(col)
            elif self._is_timeseries(s):
                detected["timeseries"].append(col)
            elif self._is_text(s):
                detected["text"].append(col)
            else:
                detected["tabular"].append(col)

        # score candidates across all modalities
        for col in df.columns:
            s = df[col]
            profile: Dict[str, Any] = {
                "column": col,
                "dtype": "text" if self._is_text(s) else ("image" if self._is_image(s) else "tabular"),
                "nan_ratio": self._nan_ratio(s),
                "unique_ratio": self._unique_ratio(s),
                "avg_len": self._avg_len(s),
                "json_ratio": self._json_ratio(s),
                "list_ratio": self._list_ratio(s),
            }

            quality = self._target_quality_score(s, col)

            # --- New Paper-Aligned Telemetry ---
            interaction_score = self._compute_interaction_score(df, col)
            uncertainty = self._compute_uncertainty(s)
            semantic_role = self._infer_semantic_role(col, s)

            semantic_summary[col] = semantic_role
            interaction_summary[col] = interaction_score
            uncertainty_summary[col] = uncertainty

            fix4_scores: Dict[str, Any] = {}

            if self.fix4_engine is not None:
                if pd.api.types.is_numeric_dtype(s):
                    problem_type = "regression"
                else:
                    problem_type = (
                        "classification_binary"
                        if s.nunique(dropna=True) == 2
                        else "classification_multiclass"
                    )

                modality_map = {
                    "text": detected.get("text", []),
                    "image": detected.get("image", []),
                    "tabular": detected.get("tabular", []),
                }

                fix4_scores = self.fix4_engine.score_target_candidates_fix4(
                    df[[col] + [c for c in df.columns if c != col]],
                    [col],
                    problem_type,
                    modality_map
                )

                if col in fix4_scores:
                    fix4_col_scores = fix4_scores[col]
                    predictability = fix4_col_scores.get("predictability_score", 0.0)
                    complementarity = fix4_col_scores.get("complementarity_score", 0.0)
                    semantic = fix4_col_scores.get("semantic_score", 0.0)
                    reason = "FIX-4: Learning-based target validation"
                    logger.debug(
                        "FIX-4 scoring [%s]: pred=%.3f, comp=%.3f, sem=%.3f",
                        col, predictability, complementarity, semantic
                    )
                else:
                    predictability = self._predictability_score(df, col)
                    complementarity = self._complementarity_score(df, col)
                    semantic = 0.0
                    reason = "Heuristic fallback"
            else:
                predictability = self._predictability_score(df, col)
                complementarity = self._complementarity_score(df, col)

            if self.fix4_engine is None or col not in fix4_scores:
                if profile["dtype"] == "text":
                    semantic = 0.0
                    if self._is_structured_label(s):
                        semantic += 0.60
                    if profile["avg_len"] < 50 and 2 <= s.nunique(dropna=True) <= 50:
                        semantic += 0.30
                    if profile["avg_len"] > 100:
                        semantic -= 0.15
                    reason = "Text semantic score"

                elif profile["dtype"] == "image":
                    semantic = 0.0
                    if self._looks_like_path(s) > 0.5:
                        semantic += 0.20
                    reason = "Image-path feature (not usually target)"

                else:
                    semantic = 0.0
                    if self._is_structured_label(s):
                        semantic += 0.35
                    if 2 <= s.nunique(dropna=True) <= 50:
                        semantic += 0.20
                    reason = "Tabular target quality"

            # 🔴 PATCH (X-S³ Scoring Overhaul)
            final = (
                0.25 * predictability +
                0.20 * complementarity +
                0.15 * semantic +
                0.15 * interaction_score +
                0.10 * 0.0 +  # cross_dataset defaults 0 here
                0.15 * (1.0 - uncertainty)
            )

            profile.update({
                "semantic_score": semantic,
                "predictability_score": predictability,
                "quality_score": quality,
                "interaction_score": interaction_score,
                "uncertainty": uncertainty,
                "semantic_role": semantic_role,
                "final_score": float(final),
                "reason": reason,
                "valid": final >= 0.20,
            })

            if profile["valid"]:
                candidates.append(profile)
            else:
                rejected.append(profile)

        def validate_override(df_local, target):
            """Enhanced validation with support for edge-case targets (multilabel, hierarchical)."""
            if target not in df_local.columns:
                return False, "Target not found"
            if df_local[target].isna().mean() > 0.8:
                return False, "Too many NaNs"
            if df_local[target].nunique(dropna=True) <= 1:
                return False, "No variance"

            try:
                series = df_local[target].dropna()
                if len(series) > 0:
                    sample = series.astype(str).head(50)
                    is_json = sample.str.contains(r'\{.*\}|\[.*\]', na=False).mean() > 0.3
                    if is_json:
                        logger.info(f"Target '{target}' detected as structured/multilabel")
                        return True, "Valid (structured target)"
            except Exception:
                pass

            return True, "Valid"

        if target_override:
            valid, msg = validate_override(df, target_override)
            if valid:
                target_col = target_override
                best_cand = next((c for c in candidates if c.get("column") == target_override), None)
                orig_conf = best_cand.get("final_score", 0.0) if best_cand else 0.0
                confidence = max(orig_conf, 0.5)
                if best_cand:
                    candidates.remove(best_cand)
                    candidates.insert(0, best_cand)
                else:
                    candidates.insert(0, {"column": target_override, "final_score": confidence, "reason": "User Override applied", "valid": True})
            else:
                logger.warning("Override rejected: %s", msg)
                target_override = None

        if not target_override:
            if not candidates:
                target_col = "Unknown"
                confidence = 0.0
            else:
                candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
                target_col = candidates[0]["column"]
                second = candidates[1]["final_score"] if len(candidates) > 1 else 0.0
                confidence = float(max(0.0, candidates[0]["final_score"] - second))

        problem_type = self._infer_problem(df, target_col)
        modalities = sorted(k for k, v in detected.items() if v)

        reasoning = {
            "selected": candidates[0] if candidates else {},
            "why_not_others": [
                {"column": c["column"], "reason": c.get("reason", ""), "score": c.get("final_score", 0)}
                for c in candidates[1:5]
            ],
            "confidence_gap": confidence,
        }

        preprocessing_hints = self._build_preprocessing_hints(modalities, target_col, problem_type, candidates)

        return IndividualSchema(
            dataset_id=dataset_id,
            detected_columns=detected,
            target_column=target_col,
            problem_type=problem_type,
            modalities=modalities,
            confidence=round(confidence, 3),
            target_profile=(candidates[0] if candidates else {}),
            reasoning=reasoning,
            candidates=candidates,
            rejected_candidates=rejected,
            preprocessing_hints=preprocessing_hints,
            selection_mode="manual_override" if target_override else "auto",
            semantic_summary=semantic_summary,
            interaction_summary=interaction_summary,
            uncertainty_summary=uncertainty_summary,
        )

    # -----------------------------------------------------------------------
    # Column-type checks
    # -----------------------------------------------------------------------

    def _is_image(self, series: pd.Series) -> bool:
        if series.dtype != "object":
            return False
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        hits = sum(
            any(ext in v.lower() for ext in self.IMAGE_EXTENSIONS)
            for v in sample
        )
        return hits > max(3, len(sample) * 0.3)

    def _is_text(self, series: pd.Series) -> bool:
        """
        Return True when the column looks like free-form text.
        Threshold: mean string length > 50 characters.
        """
        if series.dtype != "object":
            return False
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        return sample.str.len().mean() > 50

    def _is_timeseries(self, series: pd.Series) -> bool:
        if series.dtype != "object":
            return False
        sample = series.dropna().astype(str).head(30)
        if len(sample) == 0:
            return False
        return sample.str.contains(r"\[.*\]", na=False).mean() > 0.5

    # -----------------------------------------------------------------------
    # Target detection — 3-Layer system
    # -----------------------------------------------------------------------

    _ID_RE = _re_import.compile(r'(?:^|_)id(?:$|_)')

    def detect_semantic_target(self, col: pd.Series) -> bool:
        if col.dtype != "object":
            return False
        sample = col.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        json_like = sample.str.contains(r"\{.*\}", na=False).mean()
        list_like = sample.str.contains(r"\[.*\]", na=False).mean()
        avg_length = sample.str.len().mean()

        return bool(
            (json_like > 0.3 or list_like > 0.3)
            and avg_length > 10
        )

    def classify_text_column(self, series: pd.Series) -> str:
        if series.dtype != "object":
            return "not_text"
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return "not_text"

        if sample.str.contains(r"\{.*\}").mean() > 0.3:
            return "structured_label"
        if sample.str.contains(",").mean() > 0.3:
            return "multi_label"
        if series.nunique() < 20:
            return "categorical_text"
        return "free_text"

    def _semantic_analysis(self, col: pd.Series) -> Dict[str, Any]:
        n = len(col)
        unique = col.nunique(dropna=True)
        try:
            nan_ratio = float(col.isna().mean())
        except Exception:
            nan_ratio = 1.0

        info = {}
        if nan_ratio > 0.5:
            info["valid"] = False
            info["reason"] = "Too many NaNs"
            return info

        if unique <= 1:
            info["valid"] = False
            info["reason"] = "Constant column"
            return info

        info["semantic_target"] = self.detect_semantic_target(col)
        info["text_type"] = self.classify_text_column(col)

        if unique / max(n, 1) > 0.98 and col.dtype != "object":
            info["role"] = "id"
        elif col.dtype == "object" and 2 <= unique <= 20:
            info["role"] = "target_candidate"
        else:
            info["role"] = "feature"

        if col.dtype == "object":
            info["structure"] = info["text_type"]
        else:
            info["structure"] = "numeric"

        info["valid"] = True
        return info

    # -----------------------------------------------------------------------
    # PATCH 1, 2, 3, 6 — Advanced ML probes & multilabel detection
    # -----------------------------------------------------------------------
    def _compute_learnability(self, df: pd.DataFrame, target_col: str, modality_map: Any = None) -> float:
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            from sklearn.ensemble import RandomForestClassifier

            y = df[target_col].fillna("__NA__")
            X = df.drop(columns=[target_col], errors="ignore")
            X = X.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).fillna(0)

            if X.shape[1] == 0 or len(X) < 10:
                return 0.0

            from sklearn.preprocessing import LabelEncoder
            y_enc = LabelEncoder().fit_transform(y.astype(str))

            X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            return float(accuracy_score(y_val, preds))
        except Exception:
            return 0.0

    def _text_signal_score(self, s: pd.Series) -> float:
        score = 0.0
        sample = s.dropna().astype(str).head(50)
        is_structured_label = sample.str.contains(r"\{.*\}", na=False).mean() > 0.3

        if is_structured_label:
            score += 0.5
        if 2 <= s.nunique() <= 50:
            score += 0.3

        avg_len = sample.str.len().mean() if len(sample) > 0 else 0
        if avg_len > 100:
            score -= 0.2  # long text = feature, not target

        return max(0.0, min(1.0, score))

    def _image_label_quality(self, lazy_data: Any) -> float:
        classes = getattr(lazy_data, "classes", None)
        if classes and len(classes) > 1:
            return 1.0
        return 0.0

    def _cross_modal_boost(self, modalities: List[str]) -> float:
        score = 0.0
        if "text" in modalities and "tabular" in modalities:
            score += 0.2
        if "image" in modalities and "tabular" in modalities:
            score += 0.2
        return score

    @staticmethod
    def _text_predictability_score(series: pd.Series, df: pd.DataFrame, max_features: int = 500) -> float:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            texts = series.fillna("").astype(str)
            if texts.str.len().mean() < 3:
                return 0.0

            vectorizer = TfidfVectorizer(max_features=max_features)
            X = vectorizer.fit_transform(texts)
            y = texts.astype("category").cat.codes
            if y.nunique() < 2:
                return 0.0

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            return float(accuracy_score(y_val, clf.predict(X_val)))
        except Exception:
            return 0.0

    @staticmethod
    def _text_information_density(series: pd.Series) -> float:
        try:
            lengths = series.fillna("").astype(str).str.len()
            return float(lengths.std() / (lengths.mean() + 1e-5))
        except Exception:
            return 0.0

    @staticmethod
    def _image_label_separability_score(df: pd.DataFrame, label_col: str) -> float:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            X = df.drop(columns=[label_col], errors="ignore")
            y = df[label_col]
            if y.nunique() < 2:
                return 0.0

            X = X.select_dtypes(include=["number"]).fillna(0)
            if X.shape[1] == 0:
                return 0.0

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=50)
            clf.fit(X_train, y_train)
            return float(clf.score(X_val, y_val))
        except Exception:
            return 0.0

    @staticmethod
    def _infer_semantic_role(col: str, series: pd.Series) -> str:
        if series.dtype == "object":
            avg_len = series.dropna().astype(str).str.len().mean()
            if avg_len > 30:
                return "text"
        if series.nunique(dropna=True) > 0.9 * max(1, len(series)):
            return "identifier"
        if series.nunique(dropna=True) < 20:
            return "categorical_label"
        return "numeric_feature"

    @staticmethod
    def _compute_interaction_score(df: pd.DataFrame, col: str) -> float:
        scores = []
        for other in df.columns:
            if other == col:
                continue
            try:
                corr = abs(df[col].corr(df[other]))
                if not np.isnan(corr):
                    scores.append(corr)
            except Exception:
                continue
        return np.mean(scores) if scores else 0.0

    @staticmethod
    def _compute_uncertainty(series: pd.Series) -> float:
        try:
            if series.dtype == "object" or series.nunique(dropna=True) < 20:
                return float(1.0 - series.dropna().value_counts(normalize=True).max())

            mean_val = series.mean()
            return float(series.std() / (mean_val + 1e-5))
        except Exception:
            return 1.0

    @staticmethod
    def _is_multilabel(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(100)
        return bool(sample.str.contains("{").any() or sample.str.contains(":").any())

    def _get_valid_candidates(self, df: pd.DataFrame) -> List[str]:
        candidates = []
        for col in df.columns:
            if df[col].isna().mean() > 0.5:
                continue
            if df[col].nunique(dropna=True) <= 1:
                continue
            candidates.append(col)
        return candidates

    def _validate_target(self, score: TargetScore) -> Tuple[bool, str]:
        # 🔴 PATCH 1 — HARD CONSTRAINT FILTER
        if score.nan_ratio > 0.5:
            return False, "Too many NaNs"
        if score.final_score == 0.0:
            is_semantic = score.semantics.get("semantic_target", False) or score.semantics.get("is_multilabel", False)
            if not is_semantic:
                return False, "No predictive signal"
        if score.uniqueness_score > 0.99:
            return False, "Looks like ID"
        if score.uniqueness_score < 0.01:
            return False, "Constant / near constant"
        return True, "Valid"

    def _score_column(
        self,
        df: pd.DataFrame,
        col: str,
        n_rows: int,
        col_index: int = 0,
        total_cols: int = 1,
        cross_dataset_counts: Optional[Dict[str, int]] = None,
        total_datasets: int = 1,
    ) -> TargetScore:
        """
        X-S³ Engine — Unified CMTI Scoring.
        """
        series = df[col]
        name = col.lower()
        parts = _re_import.split(r'[_\s]+', name)
        last_part = parts[-1] if parts else ""

        any_match = float(any(k in name for k in self.TARGET_KEYWORDS))
        suffix_score_val = float(last_part in self.TARGET_SUFFIX_KEYWORDS)
        keyword_score = min(any_match + suffix_score_val * 0.5, 1.0)
        json_score = 0.0
        binary_penalty = 0.0
        n_unique = 0
        try:
            n_unique = series.nunique(dropna=True)
            if series.dtype == "object":
                sample = series.dropna().astype(str).head(50)
                if len(sample) > 0:
                    json_score = float(sample.str.contains(r"\{.*\}", na=False).mean())
                    vals = set(sample.str.lower().unique())
                    if vals <= self._BINARY_ATTRIBUTE_VALUES and n_unique == 2:
                        binary_penalty = 0.20
        except Exception:
            pass

        try:
            nan_ratio = float(series.isna().mean())
        except Exception:
            nan_ratio = 1.0

        uniqueness_score = max(0.0, 1.0 - (n_unique / max(n_rows, 1)))

        sem = self._semantic_analysis(series)
        is_mlabel = sem.get("text_type") == "multi_label" or json_score > 0.4

        predictability = self._predictability_score(df, col)
        complementarity = self._complementarity_score(df, col)

        cross_dataset = 0.0
        if cross_dataset_counts and total_datasets > 1:
            col_norm = name.strip()
            count = cross_dataset_counts.get(col_norm, 1)
            cross_dataset = (count - 1) / (total_datasets - 1) if count > 1 else 0.0

        degeneracy = self._degeneracy_penalty(series)
        predictability = predictability * (1.0 - degeneracy)

        dtype = "text" if self._is_text(series) else "image" if self._is_image(series) else "tabular"

        if dtype == "text":
            text_pred = self._text_predictability_score(series, df)
            density = self._text_information_density(series)
            predictability = max(predictability, text_pred)
            complementarity = (complementarity + density) / 2

        if dtype == "image":
            img_score = self._image_label_separability_score(df, col)
            predictability = max(predictability, img_score)

        if is_mlabel:
            predictability += 0.2

        quality = 0.0
        if (n_unique / max(n_rows, 1)) < 0.95:
            quality += 0.3
        if nan_ratio < 0.5:
            quality += 0.2

        is_categorical = False
        if str(series.dtype) in ("object", "category") or n_unique < 20:
            is_categorical = True
        if is_categorical:
            quality += 0.3
        if keyword_score > 0:
            quality += 0.2

        semantic_score = 0.0
        if any(k in name for k in ["code", "label", "class", "target", "diagnosis", "sentiment", "scp_codes"]):
            semantic_score += 0.5
        try:
            sample_val = series.dropna().iloc[0] if len(series.dropna()) > 0 else ""
            if isinstance(sample_val, dict) or str(sample_val).startswith("{"):
                semantic_score += 0.5
            if series.astype(str).str.contains(r"\{.*\}", na=False).mean() > 0.3:
                semantic_score += 0.5
        except Exception:
            pass
        semantic_score = min(semantic_score, 1.0)

        if any(name == k for k in ["scp_codes", "diagnosis", "label", "target", "class", "category", "sentiment"]):
            semantic_score = 1.0
            predictability += 0.25

        if dtype == "text":
            semantic_score = max(semantic_score, self._text_signal_score(series))

        if quality < 0.3:
            final_score = 0.0
            interaction_score = 0.0
            uncertainty = 1.0
            semantic_role = self._infer_semantic_role(col, series)
        else:
            learnability = self._compute_learnability(df, col, modality_map=None)
            interaction_score = self._compute_interaction_score(df, col)
            uncertainty = self._compute_uncertainty(series)
            semantic_role = self._infer_semantic_role(col, series)

            final_score = (
                0.25 * predictability +
                0.20 * complementarity +
                0.15 * semantic_score +
                0.15 * interaction_score +
                0.10 * cross_dataset +
                0.15 * (1.0 - uncertainty)
            )

            final_score += self._cross_modal_boost([dtype])

        reasons = []
        if semantic_score > 0.4:
            reasons.append("Structured / semantic label detected")
        if predictability < 0.1:
            reasons.append("Low predictability (complex or latent target)")
        if complementarity > 0.5:
            reasons.append("Provides unique signal")
        if is_categorical:
            reasons.append("Categorical target structure")
        if not reasons:
            reasons.append("General predictive target")

        explanation = [" | ".join(reasons)]
        final_score = max(0.0, final_score)

        return TargetScore(
            column=col,
            keyword_score=keyword_score,
            uniqueness_score=uniqueness_score,
            regression_score=float(pd.api.types.is_float_dtype(series) or (pd.api.types.is_numeric_dtype(series) and n_unique > 20)),
            json_score=json_score,
            predictability_score=predictability,
            complementarity_score=complementarity,
            cross_dataset_score=cross_dataset,
            degeneracy_penalty=degeneracy,
            final_score=round(final_score, 4),
            nan_ratio=nan_ratio,
            valid=True,
            reason="Valid",
            quality=quality,
            semantic_score=semantic_score,
            semantics=sem,
            explanation=explanation,
            semantic_role=semantic_role,
            interaction_score=interaction_score,
            uncertainty=uncertainty,
        )

    # ------------------------------------------------------------------
    # Layer 2 — Predictability score (Random-Forest cross-validation)
    # ------------------------------------------------------------------

    @staticmethod
    def _predictability_score(
        df: pd.DataFrame,
        column: str,
        max_rows: int = 500,
    ) -> float:
        """
        Estimate how well *other* columns predict *column* using a shallow
        RandomForest (max_depth=3) with 3-fold cross-validation.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder
            import warnings

            sub = df.head(max_rows).copy()
            y_raw = sub[column].fillna("__NA__")

            X = sub.drop(columns=[column]).select_dtypes(include=["number"])
            if len(X.columns) == 0:
                logger.debug(f"No numeric features for predictability score of {column}")
                return 0.0

            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))

            if len(le.classes_) <= 20:
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                scoring = "accuracy"
            else:
                clf = RandomForestRegressor(n_estimators=50, random_state=42)
                scoring = "r2"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(clf, X.fillna(0), y, cv=3, scoring=scoring)

            score = float(max(0.0, cv_scores.mean()))
            logger.debug(f"RandomForest predictability score for {column}: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(
                f"RandomForest predictability probe failed for column '{column}': {type(e).__name__}: {e}. "
                f"Falling back to correlation-based estimate."
            )

            try:
                X_numeric = df.drop(columns=[column]).select_dtypes(include=["number"])
                y_numeric = df[column]

                if X_numeric.shape[1] == 0:
                    return 0.0

                if not pd.api.types.is_numeric_dtype(y_numeric):
                    y_numeric = pd.factorize(y_numeric)[0]
                else:
                    y_numeric = y_numeric.fillna(y_numeric.mean())

                corrs = X_numeric.corrwith(pd.Series(y_numeric)).abs()
                max_corr = corrs.max()

                if np.isnan(max_corr):
                    logger.debug(f"Correlation fallback also failed for {column}, returning 0")
                    return 0.0

                fallback_score = float(max(0.0, max_corr))
                logger.debug(f"Correlation-based fallback for {column}: {fallback_score:.4f}")
                return fallback_score

            except Exception as e2:
                logger.error(
                    f"Both RandomForest and correlation fallback failed for {column}: {type(e2).__name__}: {e2}",
                    exc_info=False
                )
                return 0.0

    @staticmethod
    def _complementarity_score(df: pd.DataFrame, target_col: str) -> float:
        try:
            X = df.drop(columns=[target_col]).select_dtypes(include=["number"])
            y = df[target_col]

            if X.shape[1] == 0:
                return 0.0

            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            if not pd.api.types.is_numeric_dtype(y):
                y = pd.factorize(y)[0]

            corrs = X.corrwith(pd.Series(y)).abs()
            max_corr = corrs.max()
            if np.isnan(max_corr):
                return 0.0
            return float(max(0.0, 1.0 - max_corr))
        except Exception:
            return 0.0

    @staticmethod
    def _degeneracy_penalty(series: pd.Series) -> float:
        """
        Penalize targets with suspicious cardinality patterns.
        Returns: penalty in [0, 1] where 1 = completely degenerate.
        """
        try:
            n = len(series)
            unique_count = series.nunique(dropna=True)
            unique_ratio = unique_count / max(n, 1)

            if unique_count <= 1:
                return 1.0
            if unique_ratio > 0.95:
                return 0.8
            if unique_ratio < 0.02:
                return 0.7

            value_counts = series.value_counts()
            if len(value_counts) > 1:
                proportions = value_counts.values / n
                max_prop = proportions.max()
                if max_prop > 0.99:
                    return 0.6
                if max_prop > 0.95:
                    return 0.3

            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _cross_dataset_score(col: str, per_dataset: List[Dict[str, Any]]) -> float:
        try:
            count = sum(col in ds.get("detected_columns", {}).get("tabular", []) for ds in per_dataset)
            return float(count / max(len(per_dataset), 1))
        except Exception:
            return 0.0

    @staticmethod
    def _explain(c: TargetScore) -> Dict[str, float]:
        return {
            "semantic": round(0.2 * c.keyword_score, 3),
            "predictability": round(0.3 * c.predictability_score, 3),
            "complementarity": round(0.2 * c.complementarity_score, 3),
            "uniqueness": round(0.2 * c.uniqueness_score, 3),
            "penalty": round(-0.1 * c.degeneracy_penalty, 3)
        }

    @staticmethod
    def _counterfactual(candidates: List[TargetScore]) -> Dict[str, Any]:
        sorted_c = sorted(candidates, key=lambda x: x.final_score, reverse=True)
        if len(sorted_c) < 2:
            return {}
        return {
            "runner_up": sorted_c[1].column,
            "gap": round(sorted_c[0].final_score - sorted_c[1].final_score, 3)
        }

    def _detect_target(self, df: pd.DataFrame) -> Tuple[str, float, List[TargetScore]]:
        """
        Unified X-S³ Engine target inference with Hard Validation.
        Returns (best_column_name, confidence_score, candidates_list).
        """
        n_rows = max(len(df), 1)
        total_cols = len(df.columns)

        valid_candidate_names = self._get_valid_candidates(df)
        scored = []
        rejected = []

        for i, col in enumerate(df.columns):
            ts = self._score_column(df, col, n_rows, col_index=i, total_cols=total_cols)

            if col not in valid_candidate_names:
                ts.valid = False
                ts.reason = "Filtered by preliminary scan (constant or NaN-heavy)"
                rejected.append(ts)
                continue

            valid, reason = self._validate_target(ts)
            ts.valid = valid
            ts.reason = reason

            if valid:
                scored.append(ts)
            else:
                rejected.append(ts)

        if not scored:
            raise ValueError("No valid target found. Please select manually.")

        scored.sort(key=lambda s: s.final_score, reverse=True)
        rejected.sort(key=lambda s: s.final_score, reverse=True)

        all_candidates = scored + rejected
        self.last_target_candidates = all_candidates  # type: List[TargetScore]

        best = scored[0]

        if len(scored) > 1:
            best_score = scored[0].final_score
            second_score = scored[1].final_score
            best_quality = scored[0].quality
            confidence = round(
                0.5 * (best_score - second_score) + 0.5 * best_quality, 3
            )
        else:
            confidence = round(0.5 * scored[0].quality + 0.5, 3)

        return best.column, confidence, all_candidates

    # -----------------------------------------------------------------------
    # Problem-type inference
    # -----------------------------------------------------------------------

    def _infer_problem(self, df: pd.DataFrame, target: str) -> str:
        """
        Infer the ML problem type from the target column.
        """
        if target == "Unknown":
            return "unsupervised"

        s = df[target]

        if s.dtype == "object":
            sample = s.dropna().astype(str).head(50)
            if len(sample) > 0 and sample.str.contains(r"\{.*\}", na=False).mean() > 0.3:
                return "multilabel_classification"

        n_unique: int = int(s.nunique(dropna=True))

        if n_unique == 2:
            return "classification_binary"

        if 3 <= n_unique <= 20:
            return "classification_multiclass"

        if pd.api.types.is_numeric_dtype(s):
            if pd.api.types.is_float_dtype(s) or n_unique > 20:
                return "regression"
            return "classification_multiclass"

        return "classification_multiclass"

    # -----------------------------------------------------------------------
    # Tier-2 aggregation helpers
    # -----------------------------------------------------------------------

    def _aggregate_modalities(self, results: List[Dict[str, Any]]) -> List[str]:
        """Union all per-dataset modalities into a sorted list."""
        mods: set = set()
        for r in results:
            mods.update(r.get("modalities", []))
        return sorted(mods)

    def _aggregate_problem_type(self, results: List[Dict[str, Any]]) -> str:
        """
        Resolve a single global problem type.
        Regression takes priority when mixed; otherwise majority vote.
        """
        types: List[str] = [
            r["problem_type"]
            for r in results
            if r.get("problem_type", "unsupervised") != "unsupervised"
        ]

        if not types:
            return "unsupervised"

        if "regression" in types:
            return "regression"

        return max(set(types), key=types.count)

    def _collect_all_candidates(
        self,
        results: List[Dict[str, Any]],
    ) -> List[TargetScore]:
        """Merge all candidate signals across datasets into a unified ranking."""
        cmap: Dict[str, TargetScore] = {}
        for r in results:
            for c_dict in r.get("candidates", []):
                col = c_dict.get("column", "")
                if not col:
                    continue
                ts = TargetScore(
                    column=col,
                    final_score=float(c_dict.get("final_score", 0.0)),
                    nan_ratio=float(c_dict.get("nan_ratio", 0.0)),
                    valid=bool(c_dict.get("valid", True)),
                    reason=str(c_dict.get("reason", "Valid")),
                    quality=float(c_dict.get("quality_score", c_dict.get("quality", 0.0))),
                    semantic_score=float(c_dict.get("semantic_score", 0.0)),
                    semantics=c_dict.get("semantics", {}),
                    explanation=c_dict.get("explanation", []),
                    semantic_role=str(c_dict.get("semantic_role", "")),
                    interaction_score=float(c_dict.get("interaction_score", 0.0)),
                    uncertainty=float(c_dict.get("uncertainty", 0.0)),
                    predictability_score=float(c_dict.get("predictability_score", 0.0)),
                    complementarity_score=float(c_dict.get("complementarity_score", 0.0)),
                )
                ts.cross_dataset_score = self._cross_dataset_score(col, results)
                if col not in cmap or ts.final_score > cmap[col].final_score:
                    cmap[col] = ts
        return list(cmap.values())

    # -------------------------------------------------------------------
    # Cross-dataset relatedness
    # -------------------------------------------------------------------

    def _check_relatedness(
        self,
        per_dataset_results: List[Dict[str, Any]],
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Check pairwise relatedness of datasets and return groups.
        """
        n = len(per_dataset_results)
        if n <= 1:
            return [list(range(n))], {"single_dataset": True, "n_groups": 1}

        scores: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                a = per_dataset_results[i]
                b = per_dataset_results[j]

                cols_a: set = set()
                cols_b: set = set()
                for mod_cols in a.get("detected_columns", {}).values():
                    cols_a.update(mod_cols)
                for mod_cols in b.get("detected_columns", {}).values():
                    cols_b.update(mod_cols)
                union = cols_a | cols_b
                col_jaccard = len(cols_a & cols_b) / len(union) if union else 0.0

                target_match = 1.0 if (
                    a.get("target_column", "X") == b.get("target_column", "Y")
                    and a.get("target_column") != "Unknown"
                ) else 0.0

                mods_a = set(a.get("modalities", []))
                mods_b = set(b.get("modalities", []))
                mod_union = mods_a | mods_b
                mod_jaccard = (
                    len(mods_a & mods_b) / len(mod_union) if mod_union else 0.0
                )

                prob_match = (
                    1.0 if a.get("problem_type") == b.get("problem_type") else 0.0
                )

                score = (
                    0.40 * col_jaccard
                    + 0.30 * target_match
                    + 0.20 * mod_jaccard
                    + 0.10 * prob_match
                )
                scores[(i, j)] = score

        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            px, py = _find(x), _find(y)
            if px != py:
                parent[px] = py

        for (i, j), score in scores.items():
            if score >= 0.5:
                _union(i, j)

        from collections import defaultdict
        group_map: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            group_map[_find(i)].append(i)

        groups = list(group_map.values())
        report = {
            "n_datasets": n,
            "n_groups": len(groups),
            "pairwise_scores": {
                f"{i}-{j}": round(s, 3) for (i, j), s in scores.items()
            },
            "groups": groups,
        }
        return groups, report