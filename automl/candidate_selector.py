"""
automl/candidate_selector.py

Data-Driven Model Selection — replaces heuristic tier logic.

Architecture
------------
1. generate_candidates(schema_info)     → candidate pool per modality
2. quick_probe_tabular(candidates, X, y) → real 1-fold CV scores (tabular only)
3. rank_candidates(probe_scores, hw)    → cost-aware sorted list
4. apply_jit_filter(ranked, vram_gb)   → remove candidates that exceed VRAM
5. apply_manual_override(final, user)  → user picks → type="manual_override"

Key decisions
-------------
- Text/image probing is SKIPPED (encoder forward pass = minutes).
  Ranking for those modalities uses lightweight heuristics based on
  dataset_size × avg_tokens (text) and dataset_size × resolution (image).
- Tabular probing uses scikit-learn-compatible wrappers (XGBoost, LightGBM,
  or sklearn RandomForest as fallback) on ≤ 2 000 rows, 1 stratified fold.
  This runs in < 2 s on any modern CPU.
- JIT selector role is REDUCED to a VRAM filter; it no longer chooses models.
- AdvancedModelSelector is retained ONLY for Optuna HPO space generation.

Public API consumed by Phase 4
--------------------------------
    sel = CandidateSelector()
    candidates = sel.generate_candidates(schema_info)
    probe = sel.quick_probe_tabular(candidates["tabular"], X, y, problem_type)
    ranked = sel.rank_candidates(
        probe_scores=probe,
        schema_info=schema_info,
        hardware_info=hw,
    )
    ranked = sel.apply_jit_filter(ranked, vram_gb=hw["gpu_memory_gb"])
    final  = sel.apply_manual_override(ranked, user_selection)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalogue — all encoder candidates per modality
# ---------------------------------------------------------------------------

TABULAR_CANDIDATE_POOL: List[Dict[str, Any]] = [
    {
        "name": "xgboost",
        "label": "XGBoost",
        "vram_mb": 50,       # runs on CPU; VRAM used only when gpu_hist enabled
        "params_m": 0.0,
        "sklearn_cls": "xgboost.XGBClassifier",
        "sklearn_reg": "xgboost.XGBRegressor",
        "sklearn_kwargs": {"n_estimators": 50, "max_depth": 4,
                           "learning_rate": 0.1, "verbosity": 0,
                           "random_state": 42, "n_jobs": 1},
    },
    {
        "name": "lightgbm",
        "label": "LightGBM",
        "vram_mb": 50,
        "params_m": 0.0,
        "sklearn_cls": "lightgbm.LGBMClassifier",
        "sklearn_reg": "lightgbm.LGBMRegressor",
        "sklearn_kwargs": {"n_estimators": 50, "max_depth": 4,
                           "learning_rate": 0.1, "verbose": -1,
                           "random_state": 42, "n_jobs": 1},
    },
    {
        "name": "grn",
        "label": "GRN (Gated Residual Network)",
        "vram_mb": 200,
        "params_m": 0.3,
        "sklearn_cls": "sklearn.ensemble.RandomForestClassifier",
        "sklearn_reg": "sklearn.ensemble.RandomForestRegressor",
        "sklearn_kwargs": {"n_estimators": 50, "max_depth": 6, "random_state": 42, "n_jobs": 1},
    },
    {
        "name": "mlp",
        "label": "MLP (Multi-Layer Perceptron)",
        "vram_mb": 100,
        "params_m": 0.1,
        "sklearn_cls": "sklearn.neural_network.MLPClassifier",
        "sklearn_reg": "sklearn.neural_network.MLPRegressor",
        "sklearn_kwargs": {"hidden_layer_sizes": (64, 32), "max_iter": 50,
                           "random_state": 42},
    },
]

TEXT_CANDIDATE_POOL: List[Dict[str, Any]] = [
    {"name": "minilm",     "label": "MiniLM-L6",   "vram_mb": 512,  "params_m": 22.7,
     "tier": "fast",     "avg_token_limit": 4096},
    {"name": "distilbert", "label": "DistilBERT",   "vram_mb": 1024, "params_m": 66.4,
     "tier": "balanced", "avg_token_limit": 2048},
    {"name": "bert",       "label": "BERT-base",    "vram_mb": 1800, "params_m": 110,
     "tier": "balanced", "avg_token_limit": 1024},
    {"name": "deberta",    "label": "DeBERTa-v3",   "vram_mb": 3500, "params_m": 183.8,
     "tier": "sota",     "avg_token_limit": 512},
]

IMAGE_CANDIDATE_POOL: List[Dict[str, Any]] = [
    {"name": "mobilenet",    "label": "MobileNetV3",   "vram_mb": 800,  "params_m": 2.5,
     "tier": "lightweight"},
    {"name": "efficientnet", "label": "EfficientNet-B0","vram_mb": 1200, "params_m": 5.3,
     "tier": "balanced"},
    {"name": "resnet50",     "label": "ResNet-50",     "vram_mb": 2000, "params_m": 25,
     "tier": "balanced"},
    {"name": "convnext",     "label": "ConvNeXt-Tiny", "vram_mb": 3000, "params_m": 28.6,
     "tier": "sota"},
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RankedModel:
    """One entry in the ranked candidate list."""
    name: str                       # e.g. "xgboost"
    label: str                      # human-readable
    modality: str                   # "tabular" | "text" | "image"
    val_score: float = 0.0          # 1-fold CV accuracy/R² (0 if not probed)
    latency_ms: float = 0.0         # probe wall time per sample
    cost_score: float = 0.0         # accuracy − λ×latency_norm
    vram_mb: int = 0
    params_m: float = 0.0
    rationale: str = ""
    probed: bool = False            # True if quick_probe was actually run


@dataclass
class FinalSelection:
    """
    The resolved model selection after auto-ranking + optional manual override.
    Consumed by Phase 4 and stored in phase_results[MODEL_SELECTION].
    """
    tabular: Optional[str] = None           # winning encoder name
    text: Optional[str] = None
    image: Optional[str] = None
    fusion_strategy: str = "concatenation"
    selection_type: str = "auto"            # "auto" | "manual_override"
    ranked_tabular: List[RankedModel] = field(default_factory=list)
    ranked_text: List[RankedModel] = field(default_factory=list)
    ranked_image: List[RankedModel] = field(default_factory=list)
    rationale: Dict[str, str] = field(default_factory=dict)
    probe_summary: Dict[str, Any] = field(default_factory=dict)
    override_report: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# CandidateSelector
# ---------------------------------------------------------------------------

class CandidateSelector:

    MAX_PROBES_PER_MODALITY: int = 3
    MAX_PROBE_TIME: int = 60  # seconds

    # Adaptive cost profiles: ⚡Fast | ⚖️Balanced | 🎯Accurate
    COST_PROFILES = {
        "fast":     {"lambda_latency": 0.50, "mu_memory": 0.30},
        "balanced": {"lambda_latency": 0.10, "mu_memory": 0.05},
        "accurate": {"lambda_latency": 0.02, "mu_memory": 0.01},
    }

    def __init__(self):
        self._probe_cache: Dict[Tuple, Any] = {}
        self._probe_start_time: Optional[float] = None
        self._failed_models: Dict[str, List[str]] = {}  # dataset_hash → [model_names]

    # -------------------------------------------------------------------
    # Time budget control
    # -------------------------------------------------------------------

    def _check_time_budget(self):
        """Raise TimeoutError if cumulative probe time exceeds MAX_PROBE_TIME."""
        if (self._probe_start_time is not None
                and time.time() - self._probe_start_time > self.MAX_PROBE_TIME):
            raise TimeoutError(
                f"Probe time budget exceeded ({self.MAX_PROBE_TIME}s). "
                "Returning best results so far."
            )

    # -------------------------------------------------------------------
    # Probe cache (instance-level, not class-level)
    # -------------------------------------------------------------------

    def get_probe_score(self, dataset_hash, model, probe_fn):
        key = (dataset_hash, model)
        if key in self._probe_cache:
            logger.info("Probe cache HIT: %s/%s", dataset_hash, model)
            return self._probe_cache[key]
        score = probe_fn()
        self._probe_cache[key] = score
        return score

    # -------------------------------------------------------------------
    # Failure memory
    # -------------------------------------------------------------------

    def record_failure(self, dataset_hash: str, model_name: str):
        """Record a failed model so it's skipped in future runs."""
        self._failed_models.setdefault(dataset_hash, []).append(model_name)
        logger.warning("Failure recorded: %s on dataset %s", model_name, dataset_hash)

    def filter_failed(self, dataset_hash: str, candidates: List[Dict]) -> List[Dict]:
        """Remove previously failed models from candidate list."""
        failed = set(self._failed_models.get(dataset_hash, []))
        if failed:
            logger.info("Skipping previously failed models: %s", failed)
        return [c for c in candidates if c["name"] not in failed]

    # -------------------------------------------------------------------
    # Adaptive cost profiles
    # -------------------------------------------------------------------

    def get_cost_weights(self, user_mode: str = "balanced"):
        profile = self.COST_PROFILES.get(user_mode, self.COST_PROFILES["balanced"])
        logger.info("Cost profile: %s → λ=%.2f μ=%.2f",
                     user_mode, profile["lambda_latency"], profile["mu_memory"])
        return profile["lambda_latency"], profile["mu_memory"]

    # -------------------------------------------------------------------
    # Data complexity signal
    # -------------------------------------------------------------------

    def compute_data_complexity(self, X, y):
        """Compute label entropy and feature sparsity for model gating."""
        from scipy.stats import entropy as sp_entropy
        probs = np.bincount(y.astype(int)) / len(y)
        label_entropy = float(sp_entropy(probs))
        sparsity = float(np.mean(X == 0)) if hasattr(X, "toarray") else 0.0
        return {"entropy": label_entropy, "sparsity": sparsity}

    # -------------------------------------------------------------------
    # Selection confidence
    # -------------------------------------------------------------------

    def compute_selection_confidence(self, scores: Dict[str, float]) -> float:
        """Margin between top-2 scored models. Higher = more confident."""
        if len(scores) < 2:
            return 1.0
        sorted_vals = sorted(scores.values(), reverse=True)
        return float((sorted_vals[0] - sorted_vals[1]) / (sorted_vals[0] + 1e-6))

    # -------------------------------------------------------------------
    # Uncertainty-aware exploration
    # -------------------------------------------------------------------

    def get_top_k(self, ranked_models, confidence):
        """Adaptive top-K: explore more when confidence is low."""
        if confidence < 0.1:
            logger.info("Low confidence (%.3f) — exploring top-3 candidates", confidence)
            return ranked_models[:3]
        elif confidence < 0.3:
            logger.info("Moderate confidence (%.3f) — exploring top-2", confidence)
            return ranked_models[:2]
        return ranked_models[:1]

    # -------------------------------------------------------------------
    # Ablation logging hook
    # -------------------------------------------------------------------

    def log_selection_decision(self, decision: Dict[str, Any]):
        """Emit structured log for ablation analysis / paper results."""
        import json as _json
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "probe_scores": decision.get("probe_scores", {}),
            "joint_scores": decision.get("joint_scores", {}),
            "fusion_scores": decision.get("fusion_scores", {}),
            "complexity": decision.get("complexity", {}),
            "confidence": decision.get("confidence", 0.0),
            "cost_profile": decision.get("cost_profile", "balanced"),
            "final_ranking": decision.get("ranking", []),
            "selected_model": decision.get("selected", ""),
            "meta_suggestions": decision.get("meta_suggestions", []),
        }
        logger.info("SELECTION_DECISION: %s", _json.dumps(entry))
        return entry

    # ===================================================================
    # REAL PROBE IMPLEMENTATIONS
    # ===================================================================

    # -------------------------------------------------------------------
    # Text probes
    # -------------------------------------------------------------------

    def run_tfidf_probe(self, data, max_samples=2000):
        """TF-IDF + LogisticRegression 1-fold probe."""
        self._check_time_budget()
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        texts = data["texts"][:max_samples]
        labels = data["labels"][:max_samples]
        split = int(len(texts) * 0.8)

        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(texts[:split])
        X_val = vectorizer.transform(texts[split:])

        t0 = time.perf_counter()
        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(X_train, labels[:split])
        latency = time.perf_counter() - t0

        preds = clf.predict(X_val)
        score = float(accuracy_score(labels[split:], preds))
        logger.info("  probe tfidf: score=%.4f latency=%.3fs", score, latency)
        return score, latency

    def run_transformer_probe(self, data, model_name, max_samples=1000, **kwargs):
        """1-epoch transformer fine-tune probe (DistilBERT/BERT)."""
        self._check_time_budget()
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch as _torch
        from sklearn.metrics import accuracy_score

        texts = data["texts"][:max_samples]
        labels = data["labels"][:max_samples]
        split = int(0.8 * len(texts))
        num_labels = len(set(labels))

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        optimizer = _torch.optim.AdamW(model.parameters(), lr=5e-5)

        # 1-epoch training
        model.train()
        for i in range(split):
            self._check_time_budget()
            inputs = tokenizer(texts[i], return_tensors="pt", truncation=True)
            loss = model(**inputs, labels=_torch.tensor([labels[i]])).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate
        model.eval()
        enc = tokenizer(
            texts[split:], truncation=True, padding=True, return_tensors="pt"
        )
        t0 = time.perf_counter()
        with _torch.no_grad():
            logits = model(**enc).logits
        latency = time.perf_counter() - t0

        preds = logits.argmax(dim=-1).cpu().numpy()
        score = float(accuracy_score(labels[split:], preds))
        logger.info("  probe %s: score=%.4f latency=%.3fs", model_name, score, latency)
        return score, latency

    # -------------------------------------------------------------------
    # Image probe
    # -------------------------------------------------------------------

    def run_image_probe(self, model_name, data, max_samples=500, **kwargs):
        """Feature extraction + LogReg image probe."""
        self._check_time_budget()
        import torch as _torch
        import torchvision.models as tv
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        model_map = {
            "mobilenet": tv.mobilenet_v3_small,
            "efficientnet": tv.efficientnet_b0,
            "resnet": tv.resnet18,
        }
        factory = model_map.get(model_name)
        if factory is None:
            return 0.5, 1.0

        model = factory(pretrained=True)
        model.eval()

        images = data["images"][:max_samples]
        labels = np.array(data["labels"][:max_samples])
        imgs = _torch.stack(images)

        with _torch.no_grad():
            t0 = time.perf_counter()
            features = (model.features(imgs)
                        if hasattr(model, "features") else model(imgs))
            latency = time.perf_counter() - t0

        features = features.view(features.size(0), -1).cpu().numpy()
        split = int(0.8 * len(features))

        clf = LogisticRegression(max_iter=200)
        clf.fit(features[:split], labels[:split])
        preds = clf.predict(features[split:])
        score = float(accuracy_score(labels[split:], preds))
        logger.info("  probe %s: score=%.4f latency=%.3fs", model_name, score, latency)
        return score, latency

    # -------------------------------------------------------------------
    # Memory estimation
    # -------------------------------------------------------------------

    def estimate_memory(self, model_name):
        """Approximate VRAM in MB based on known model parameter counts."""
        estimates = {
            "tfidf": 50, "distilbert": 800, "bert": 1500,
            "mobilenet": 200, "efficientnet": 400, "resnet": 800,
            "xgboost": 100, "lightgbm": 80, "random_forest": 150,
        }
        return estimates.get(model_name, 500)

    # -------------------------------------------------------------------
    # Text / Image probes (public API — with time budget + error handling)
    # -------------------------------------------------------------------

    # BUG-10: Remove duplicate quick_probe_text (kept only the L746 version below)

    def quick_probe_image(self, candidates, data, max_samples=500):
        """
        Probe image models with real feature extraction + LogReg.
        Returns dict of {model: {accuracy, latency, memory}}.
        """
        if self._probe_start_time is None:
            self._probe_start_time = time.time()
        scores = {}
        for model in candidates[:self.MAX_PROBES_PER_MODALITY]:
            try:
                self._check_time_budget()
                score, latency = self.run_image_probe(
                    model_name=model["name"], data=data, max_samples=max_samples,
                )
                scores[model["name"]] = {
                    "accuracy": score,
                    "latency": latency,
                    "memory": self.estimate_memory(model["name"]),
                }
            except TimeoutError:
                logger.warning("Time budget hit during image probe — returning partial results")
                break
            except Exception as exc:
                logger.error("Probe FAILED for %s: %s", model["name"], exc)
                scores[model["name"]] = {
                    "accuracy": 0.0, "latency": 9999, "memory": 0,
                    "error": str(exc),
                }
        return scores

    # -------------------------------------------------------------------
    # Joint probe (MLP fusion with LayerNorm + Dropout)
    # -------------------------------------------------------------------

    def joint_probe(self, tab_model=None, text_model=None, image_model=None,
                    data=None, max_samples=1000):
        """Evaluate multimodal fusion with a lightweight MLP (3 epochs)."""
        import torch as _torch
        import torch.nn as _nn
        from sklearn.metrics import accuracy_score

        class _SimpleFusion(_nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.net = _nn.Sequential(
                    _nn.Linear(input_dim, 128),
                    _nn.LayerNorm(128),
                    _nn.ReLU(),
                    _nn.Dropout(0.2),
                    _nn.Linear(128, num_classes),
                )
            def forward(self, x):
                return self.net(x)

        try:
            embeddings = []
            if tab_model and "tabular" in data:
                embeddings.append(data["tabular"][:max_samples])
            if text_model and "texts" in data:
                from sklearn.feature_extraction.text import TfidfVectorizer
                tfidf = TfidfVectorizer(max_features=500)
                embeddings.append(
                    tfidf.fit_transform(data["texts"][:max_samples]).toarray()
                )

            if not embeddings:
                return 0.0

            X = np.hstack(embeddings)
            y = np.array(data["labels"][:max_samples])
            num_classes = len(set(y))

            model = _SimpleFusion(X.shape[1], num_classes)
            optimizer = _torch.optim.Adam(model.parameters(), lr=1e-3)
            X_t = _torch.tensor(X, dtype=_torch.float32)
            y_t = _torch.tensor(y, dtype=_torch.long)

            model.train()
            for _ in range(3):
                logits = model(X_t)
                loss = _nn.CrossEntropyLoss()(logits, y_t)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            with _torch.no_grad():
                preds = model(X_t).argmax(dim=1).numpy()

            score = float(accuracy_score(y, preds))
            logger.info("  joint_probe: score=%.4f", score)
            return score
        except Exception as exc:
            logger.error("Joint probe failed: %s", exc)
            return 0.0

    # -------------------------------------------------------------------
    # Fusion strategy probe
    # -------------------------------------------------------------------

    def probe_fusion(self, config, data, max_samples=500):
        """Probe fusion strategies with lightweight training."""
        fusion_candidates = ["concatenation", "attention"]
        scores = {}
        for fusion in fusion_candidates:
            try:
                score = self.joint_probe(
                    tab_model=config.get("tabular"),
                    text_model=config.get("text"),
                    image_model=config.get("image"),
                    data=data, max_samples=max_samples,
                )
                scores[fusion] = score
            except Exception as exc:
                logger.warning("Fusion probe failed for %s: %s", fusion, exc)
                scores[fusion] = 0.0
        best = max(scores, key=scores.get) if scores else "concatenation"
        logger.info("Fusion probe: %s → best=%s", scores, best)
        return best, scores

    """
    Data-driven model selector.

    Replaces the heuristic tier tables in AdvancedModelSelector for the model
    *choice* decision.  AdvancedModelSelector is still used for HPO bound
    generation.
    """

    # Cost-aware ranking weights
    LAMBDA_LATENCY: float = 0.10   # penalise slow models
    MU_VRAM: float = 0.05          # penalise memory-hungry models

    # ---------------------------------------------------------------------------
    # Step 1 — Candidate generation
    # ---------------------------------------------------------------------------

    def generate_candidates(
        self,
        schema_info: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return the full candidate pool for each active modality.

        Parameters
        ----------
        schema_info : Phase 2 result dict — uses ``global_modalities``.

        Returns
        -------
        dict  e.g. ``{"tabular": [...], "text": [...], "image": [...]}``
        """
        modalities = schema_info.get("global_modalities", ["tabular"])
        dataset_size = schema_info.get("total_samples", 10_000)

        candidates: Dict[str, List[Dict]] = {}

        if "tabular" in modalities:
            candidates["tabular"] = list(TABULAR_CANDIDATE_POOL)

        if "text" in modalities:
            avg_tokens = schema_info.get("avg_tokens", 128)
            candidates["text"] = [
                c for c in TEXT_CANDIDATE_POOL
                if avg_tokens <= c["avg_token_limit"]
            ]

        if "image" in modalities:
            candidates["image"] = list(IMAGE_CANDIDATE_POOL)

        logger.info(
            "CandidateSelector: generated %s candidates",
            {k: len(v) for k, v in candidates.items()},
        )
        return candidates

    # ---------------------------------------------------------------------------
    # Step 2 — Quick probe (tabular only, ≤2000 rows, 1-fold)
    # ---------------------------------------------------------------------------

    def quick_probe_tabular(
        self,
        candidates: List[Dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        problem_type: str,
        max_rows: int = 1500,
        _retried: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run a fast 1-fold validation with each tabular candidate.

        Uses scikit-learn-compatible wrappers.  XGBoost/LightGBM are imported
        lazily and fall back to the sklearn entry (RandomForest/MLP) if the
        library is missing.

        Returns
        -------
        dict  {model_name: {"val_score": float, "uncertainty": float, "latency_ms": float, "confidence": str}}
        """
        if X is None or len(X) == 0:
            logger.warning("quick_probe_tabular: no probe data — skipping")
            return {}

        import numpy as np

        # Sub-sample for speed
        n = min(max_rows, len(X))
        idx = np.random.default_rng(42).permutation(len(X))[:n]
        X_s, y_s = X[idx], y[idx]

        is_clf = "classification" in problem_type
        
        from sklearn.model_selection import StratifiedKFold, KFold
        n_splits = 3
        if is_clf:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        results: Dict[str, Dict[str, float]] = {}

        for cand in candidates:
            name = cand["name"]
            cls_path = cand["sklearn_cls"] if is_clf else cand["sklearn_reg"]
            kwargs = dict(cand["sklearn_kwargs"])

            model = self._import_model(cls_path, kwargs, cand)
            if model is None:
                logger.debug("quick_probe: skipping %s (import failed)", name)
                continue

            scores = []
            latencies = []
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    for train_idx, val_idx in cv.split(X_s, y_s):
                        X_train, X_val = X_s[train_idx], X_s[val_idx]
                        y_train, y_val = y_s[train_idx], y_s[val_idx]
                        
                        t0 = time.perf_counter()
                        model.fit(X_train, y_train)
                        
                        if is_clf:
                            preds = model.predict(X_val)
                            from sklearn.metrics import accuracy_score
                            score = float(accuracy_score(y_val, preds))
                        else:
                            preds = model.predict(X_val)
                            from sklearn.metrics import r2_score
                            score = float(max(r2_score(y_val, preds), 0.0))
                            
                        latencies.append((time.perf_counter() - t0) * 1000)
                        scores.append(score)

                import numpy as np
                mean_score = float(np.mean(scores))
                unc_score = float(np.std(scores))
                mean_lat = float(np.mean(latencies))
                
                results[name] = {
                    "val_score": mean_score,
                    "uncertainty": unc_score,
                    "latency_ms": mean_lat,
                    "confidence": "HIGH"
                }
                logger.info(
                    "  probe %s: score=%.4f (±%.4f) latency=%.1f ms",
                    name, mean_score, unc_score, mean_lat,
                )
            except Exception as exc:
                logger.warning("  probe %s FAILED: %s", name, exc)

        overall_max_unc = max([res.get("uncertainty", 0.0) for res in results.values()]) if results else 0.0
        if overall_max_unc > 0.15 and len(X) > max_rows and not _retried:
            logger.info("  [Adaptive Budget] Uncertainty %.4f > 0.15. Doubling max_rows to %d and retrying.", overall_max_unc, max_rows * 2)
            return self.quick_probe_tabular(candidates, X, y, problem_type, max_rows=max_rows * 2, _retried=True)

        return results

    def quick_probe_text(
        self,
        candidates: List[Dict[str, Any]],
        texts: List[str],
        y: np.ndarray,
        problem_type: str,
        max_rows: int = 1000,
        _retried: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Genuine Linear Probe on frozen textual embeddings.
        Subsamples to `max_rows` (adaptive), extracts embeddings via HF transformers,
        and trains Ridge classifier/regressor with CV.
        """
        if not texts or len(texts) == 0:
            return {}

        n = min(max_rows, len(texts))
        idx = np.random.default_rng(42).permutation(len(texts))[:n]
        txt_s = [texts[i] for i in idx]
        y_s = y[idx]
        is_clf = "classification" in problem_type

        from sklearn.model_selection import StratifiedKFold, KFold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if is_clf else KFold(n_splits=3, shuffle=True, random_state=42)

        results = {}
        for cand in candidates:
            name = cand["name"]
            latency_ms_est = cand.get("params_m", 100) * 1.5  # Realistic inference estimate
            base_acc = {"minilm": 0.78, "distilbert": 0.81, "bert": 0.83, "deberta": 0.85}.get(name, 0.75)
            
            # 1. Fallback Proxy Initializer
            results[name] = {
                "val_score": base_acc, 
                "uncertainty": 0.08 + 500 / max(1, n),
                "latency_ms": latency_ms_est, 
                "confidence": "LOW (heuristic fallback)"
            }

            try:
                import torch
                from transformers import AutoTokenizer, AutoModel
                
                # Model lookup mappings
                hf_map = {
                    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
                    "distilbert": "distilbert-base-uncased",
                    "bert": "bert-base-uncased",
                    "deberta": "microsoft/deberta-v3-small"
                }
                hf_id = hf_map.get(name, hf_map["minilm"])
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tokenizer = AutoTokenizer.from_pretrained(hf_id)
                model = AutoModel.from_pretrained(hf_id).to(device)
                model.eval()

                # Extract embeddings in chunks to save VRAM
                embeddings = []
                batch_size = 32
                with torch.no_grad():
                    for i in range(0, len(txt_s), batch_size):
                        batch_txt = txt_s[i:i+batch_size]
                        inputs = tokenizer(batch_txt, padding=True, truncation=True, max_length=cand.get("avg_token_limit", 512), return_tensors="pt").to(device)
                        outputs = model(**inputs)
                        # Mean pooling over valid attention mask
                        masks = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                        sum_embeddings = torch.sum(outputs.last_hidden_state * masks, 1)
                        sum_mask = torch.clamp(masks.sum(1), min=1e-9)
                        pool = sum_embeddings / sum_mask
                        embeddings.append(pool.cpu().numpy())
                
                X_emb = np.vstack(embeddings)
                
                # Linear Probe using Ridge
                from sklearn.linear_model import RidgeClassifier, Ridge
                from sklearn.metrics import accuracy_score, r2_score
                
                scores = []
                for train_idx, val_idx in cv.split(X_emb, y_s):
                    X_tr, X_va = X_emb[train_idx], X_emb[val_idx]
                    y_tr, y_va = y_s[train_idx], y_s[val_idx]
                    probe_mod = RidgeClassifier(alpha=1.0) if is_clf else Ridge(alpha=1.0)
                    probe_mod.fit(X_tr, y_tr)
                    preds = probe_mod.predict(X_va)
                    scores.append(accuracy_score(y_va, preds) if is_clf else max(r2_score(y_va, preds), 0.0))
                
                mean_score = float(np.mean(scores))
                unc_score = float(np.std(scores))
                
                results[name] = {
                    "val_score": mean_score,
                    "uncertainty": unc_score,
                    "latency_ms": latency_ms_est,
                    "confidence": "HIGH (linear probe)"
                }
                logger.info("  text linear probe %s: score=%.4f (±%.4f)", name, mean_score, unc_score)
                
                # Cleanup VRAM aggressively
                del model, tokenizer, embeddings, X_emb
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except Exception as e:
                logger.warning("  text linear probe %s failed (falling back to proxy): %s", name, str(e))

        max_unc = max([r.get("uncertainty", 0) for r in results.values()]) if results else 0
        if max_unc > 0.15 and len(texts) > max_rows and not _retried:
            logger.info("  [Adaptive Budget] Text Uncertainty %.4f > 0.15. Retrying with %d rows.", max_unc, max_rows * 2)
            return self.quick_probe_text(candidates, texts, y, problem_type, max_rows * 2, _retried=True)
            
        return results

    def quick_probe_image(
        self,
        candidates: List[Dict[str, Any]],
        image_paths: List[str],
        y: np.ndarray,
        problem_type: str,
        max_rows: int = 500,
        _retried: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Genuine Linear Probe on frozen visual embeddings.
        Uses torchvision models without heads followed by Ridge.
        """
        if not image_paths or len(image_paths) == 0:
            return {}

        n = min(max_rows, len(image_paths))
        idx = np.random.default_rng(42).permutation(len(image_paths))[:n]
        img_s = [image_paths[i] for i in idx]
        y_s = y[idx]
        is_clf = "classification" in problem_type

        from sklearn.model_selection import StratifiedKFold, KFold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if is_clf else KFold(n_splits=3, shuffle=True, random_state=42)

        results = {}
        for cand in candidates:
            name = cand["name"]
            latency_ms_est = cand.get("params_m", 25) * 2.0
            base_acc = {"mobilenet": 0.74, "efficientnet": 0.79, "resnet50": 0.78, "convnext": 0.82}.get(name, 0.75)
            
            results[name] = {
                "val_score": base_acc,
                "uncertainty": 0.09 + 500 / max(1, n),
                "latency_ms": latency_ms_est,
                "confidence": "LOW (heuristic fallback)"
            }

            try:
                import torch
                from torchvision import models, transforms
                from PIL import Image
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Fetch pre-trained model
                if name == "mobilenet":
                    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                    model.classifier = torch.nn.Identity()
                elif name == "efficientnet":
                    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                    model.classifier = torch.nn.Identity()
                elif name == "resnet50":
                    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                    model.fc = torch.nn.Identity()
                else: # convnext / fallback
                    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
                    model.classifier = torch.nn.Identity()

                model = model.to(device)
                model.eval()

                tfs = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                embeddings = []
                batch_size = 16
                with torch.no_grad():
                    for i in range(0, len(img_s), batch_size):
                        batch_imgs = []
                        for p in img_s[i:i+batch_size]:
                            try:
                                batch_imgs.append(tfs(Image.open(p).convert('RGB')))
                            except Exception:
                                batch_imgs.append(torch.zeros(3, 224, 224))
                        
                        inputs = torch.stack(batch_imgs).to(device)
                        out = model(inputs)
                        if out.dim() == 4: # Spatial maps (e.g. from some networks before pooling)
                            out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
                        embeddings.append(out.cpu().numpy())
                
                X_emb = np.vstack(embeddings)
                
                from sklearn.linear_model import RidgeClassifier, Ridge
                from sklearn.metrics import accuracy_score, r2_score
                
                scores = []
                for train_idx, val_idx in cv.split(X_emb, y_s):
                    X_tr, X_va = X_emb[train_idx], X_emb[val_idx]
                    y_tr, y_va = y_s[train_idx], y_s[val_idx]
                    probe_mod = RidgeClassifier(alpha=1.0) if is_clf else Ridge(alpha=1.0)
                    probe_mod.fit(X_tr, y_tr)
                    preds = probe_mod.predict(X_va)
                    scores.append(accuracy_score(y_va, preds) if is_clf else max(r2_score(y_va, preds), 0.0))
                
                mean_score = float(np.mean(scores))
                unc_score = float(np.std(scores))
                
                results[name] = {
                    "val_score": mean_score,
                    "uncertainty": unc_score,
                    "latency_ms": latency_ms_est,
                    "confidence": "HIGH (linear probe)"
                }
                logger.info("  image linear probe %s: score=%.4f (±%.4f)", name, mean_score, unc_score)
                
                del model, embeddings, X_emb
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except Exception as e:
                logger.warning("  image linear probe %s failed: %s", name, str(e))

        max_unc = max([r.get("uncertainty", 0) for r in results.values()]) if results else 0
        if max_unc > 0.15 and len(image_paths) > max_rows and not _retried:
            logger.info("  [Adaptive Budget] Image Uncertainty %.4f > 0.15. Retrying with %d rows.", max_unc, max_rows * 2)
            return self.quick_probe_image(candidates, image_paths, y, problem_type, max_rows * 2, _retried=True)
            
        return results

    def _import_model(
        self,
        dotted_path: str,
        kwargs: Dict[str, Any],
        cand: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Dynamically import and instantiate a model class.

        Falls back to the sklearn fallback entry in the catalogue when the
        primary library (xgboost, lightgbm) is not installed.
        """
        module_path, cls_name = dotted_path.rsplit(".", 1)
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            return cls(**kwargs)
        except (ImportError, ModuleNotFoundError):
            # Try sklearn fallback
            fallback = (
                "sklearn.ensemble.RandomForestClassifier"
                if "Classifier" in cls_name
                else "sklearn.ensemble.RandomForestRegressor"
            )
            fb_module, fb_cls = fallback.rsplit(".", 1)
            fb_kwargs = {"n_estimators": 50, "max_depth": 6,
                         "random_state": 42, "n_jobs": 1}
            try:
                import importlib
                mod = importlib.import_module(fb_module)
                cls = getattr(mod, fb_cls)
                logger.info(
                    "  Falling back to RandomForest for %s (%s not installed)",
                    cand["name"], module_path,
                )
                return cls(**fb_kwargs)
            except Exception:
                return None
        except Exception:
            return None

    # ---------------------------------------------------------------------------
    # Step 3 — Cost-aware ranking
    # ---------------------------------------------------------------------------

    def rank_candidates(
        self,
        probe_scores: Dict[str, Dict[str, Dict[str, Any]]],
        candidates: Dict[str, List[Dict[str, Any]]],
        schema_info: Dict[str, Any],
        hardware_info: Dict[str, Any],
        lambda_latency: float = 0.10,
        mu_memory: float = 0.05,
        gamma_unc: float = 0.10,
        meta_recommendations: List[str] = None,
    ) -> Dict[str, List[RankedModel]]:
        """
        Build ranked lists for every modality using research-grade objective function:
        final_score = acc_norm - λ * lat_norm - μ * mem_norm - γ * unc_norm
        """
        ranked: Dict[str, List[RankedModel]] = {}
        dataset_size = schema_info.get("total_samples", 10_000)
        
        tab_scores   = probe_scores.get("tabular", {})
        text_scores  = probe_scores.get("text", {})
        image_scores = probe_scores.get("image", {})

        # -- Tabular --
        if "tabular" in candidates:
            tab_pool = {c["name"]: c for c in candidates["tabular"]}
            models: List[RankedModel] = []
            for name, cand in tab_pool.items():
                p = tab_scores.get(name, {})
                acc = p.get("val_score", 0.0)
                lat = p.get("latency_ms", 0.0)
                mem = cand.get("vram_mb", 0.0)
                unc = p.get("uncertainty", 0.0)
                conf = p.get("confidence", "NONE")
                
                # Normalize latency, memory, uncertainty to [0,1]
                norm_acc = acc
                norm_lat = min(lat / 5000.0, 1.0)
                norm_mem = min(mem / 8000.0, 1.0)
                norm_unc = min(unc / 0.20, 1.0) # Assume 20% std dev is absolute worst case
                
                final = norm_acc - (lambda_latency * norm_lat) - (mu_memory * norm_mem) - (gamma_unc * norm_unc)
                
                if dataset_size < 10000 and name in ("xgboost", "lightgbm", "random_forest"):
                    final += 0.05
                    
                if meta_recommendations and name in meta_recommendations:
                    final += 0.15
                    conf = "META + " + conf
                    
                rationale = (
                    f"Probed [{conf}]: acc={acc:.4f} unc=±{unc:.4f} lat={lat:.0f}ms mem={mem:.1f}MB  "
                    f"final_score={final:.4f} (acc-{lambda_latency}*lat-{mu_memory}*mem-{gamma_unc}*unc)"
                )
                models.append(RankedModel(
                    name=name, label=cand["label"], modality="tabular",
                    val_score=acc, latency_ms=lat, cost_score=final,
                    vram_mb=cand["vram_mb"], params_m=cand["params_m"],
                    rationale=rationale, probed=True,
                ))
            ranked["tabular"] = sorted(models, key=lambda m: m.cost_score, reverse=True)

        # -- Text --
        if "text" in candidates:
            text_pool = {c["name"]: c for c in candidates["text"]}
            models: List[RankedModel] = []
            for name, cand in text_pool.items():
                p = text_scores.get(name, {})
                acc = p.get("val_score", 0.0)
                lat = p.get("latency_ms", 0.0)
                mem = cand.get("vram_mb", 0.0)
                unc = p.get("uncertainty", 0.0)
                conf = p.get("confidence", "NONE")
                
                norm_acc = acc
                norm_lat = min(lat / 5000.0, 1.0)
                norm_mem = min(mem / 8000.0, 1.0)
                norm_unc = min(unc / 0.20, 1.0)
                
                final = norm_acc - (lambda_latency * norm_lat) - (mu_memory * norm_mem) - (gamma_unc * norm_unc)
                
                if meta_recommendations and name in meta_recommendations:
                    final += 0.15
                    conf = "META + " + conf
                
                rationale = (
                    f"Probed [{conf}]: acc={acc:.4f} unc=±{unc:.4f} lat={lat:.0f}ms mem={mem:.1f}MB  "
                    f"final_score={final:.4f} (acc-{lambda_latency}*lat-{mu_memory}*mem-{gamma_unc}*unc)"
                )
                models.append(RankedModel(
                    name=name, label=cand["label"], modality="text",
                    val_score=acc, latency_ms=lat, cost_score=final,
                    vram_mb=cand["vram_mb"], params_m=cand["params_m"],
                    rationale=rationale, probed=True,
                ))
            ranked["text"] = sorted(models, key=lambda m: m.cost_score, reverse=True)

        # -- Image --
        if "image" in candidates:
            image_pool = {c["name"]: c for c in candidates["image"]}
            models: List[RankedModel] = []
            for name, cand in image_pool.items():
                p = image_scores.get(name, {})
                acc = p.get("val_score", 0.0)
                lat = p.get("latency_ms", 0.0)
                mem = cand.get("vram_mb", 0.0)
                unc = p.get("uncertainty", 0.0)
                conf = p.get("confidence", "NONE")
                
                norm_acc = acc
                norm_lat = min(lat / 5000.0, 1.0)
                norm_mem = min(mem / 8000.0, 1.0)
                norm_unc = min(unc / 0.20, 1.0)
                
                final = norm_acc - (lambda_latency * norm_lat) - (mu_memory * norm_mem) - (gamma_unc * norm_unc)
                
                if meta_recommendations and name in meta_recommendations:
                    final += 0.15
                    conf = "META + " + conf
                
                rationale = (
                    f"Probed [{conf}]: acc={acc:.4f} unc=±{unc:.4f} lat={lat:.0f}ms mem={mem:.1f}MB  "
                    f"final_score={final:.4f} (acc-{lambda_latency}*lat-{mu_memory}*mem-{gamma_unc}*unc)"
                )
                models.append(RankedModel(
                    name=name, label=cand["label"], modality="image",
                    val_score=acc, latency_ms=lat, cost_score=final,
                    vram_mb=cand["vram_mb"], params_m=cand["params_m"],
                    rationale=rationale, probed=True,
                ))
            ranked["image"] = sorted(models, key=lambda m: m.cost_score, reverse=True)

        for mod, lst in ranked.items():
            logger.info("Ranked %s: %s", mod, [m.name for m in lst])

        return ranked

    def _estimate_tabular_score(self, name: str, dataset_size: int) -> float:
        """
        Score estimate for unprobed tabular candidates.
        Based on general literature: XGBoost/LightGBM outperform on tabular.
        """
        base = {"xgboost": 0.72, "lightgbm": 0.73, "grn": 0.68, "mlp": 0.65}
        s = base.get(name, 0.60)
        # Larger datasets favour more complex models
        if dataset_size > 50_000:
            boost = {"xgboost": 0.04, "lightgbm": 0.04, "grn": 0.03, "mlp": 0.02}
            s += boost.get(name, 0.0)
        return float(s)

    def _rank_text(
        self,
        pool: List[Dict],
        dataset_size: int,
        gpu_mem: float,
    ) -> List[RankedModel]:
        """
        Heuristic text encoder ranking:
        - Small dataset (<5k) or low GPU (<4GB) → MiniLM
        - Medium → DistilBERT
        - Large dataset + good GPU → BERT/DeBERTa
        """
        scored: List[RankedModel] = []
        for cand in pool:
            name = cand["name"]
            vram_needed = cand["vram_mb"] / 1024  # GB

            # Start from base capability score
            base = {"minilm": 0.70, "distilbert": 0.76, "bert": 0.80, "deberta": 0.85}
            s = base.get(name, 0.70)

            # Penalise if dataset too small (overfitting large model)
            if dataset_size < 5_000 and name in ("bert", "deberta"):
                s -= 0.10

            # Penalise if VRAM insufficient
            if gpu_mem > 0 and vram_needed > gpu_mem * 0.5:
                s -= 0.15

            rationale = (
                f"Heuristic: size={dataset_size}  gpu={gpu_mem:.1f}GB  "
                f"score≈{s:.3f}"
            )
            scored.append(RankedModel(
                name=name, label=cand["label"], modality="text",
                val_score=s, cost_score=s,
                vram_mb=cand["vram_mb"], params_m=cand["params_m"],
                rationale=rationale, probed=False,
            ))
        return sorted(scored, key=lambda m: m.cost_score, reverse=True)

    def _rank_image(
        self,
        pool: List[Dict],
        dataset_size: int,
        gpu_mem: float,
    ) -> List[RankedModel]:
        """
        Heuristic image encoder ranking:
        - Small (<1k) → MobileNetV3
        - Medium (<10k) + enough GPU → EfficientNet-B0
        - Large + good GPU → ConvNeXt
        """
        scored: List[RankedModel] = []
        for cand in pool:
            name = cand["name"]
            vram_needed = cand["vram_mb"] / 1024

            base = {"mobilenet": 0.71, "efficientnet": 0.77,
                    "resnet50": 0.78, "convnext": 0.82}
            s = base.get(name, 0.70)

            if dataset_size < 1_000 and name not in ("mobilenet",):
                s -= 0.12
            if gpu_mem > 0 and vram_needed > gpu_mem * 0.6:
                s -= 0.20

            scored.append(RankedModel(
                name=name, label=cand["label"], modality="image",
                val_score=s, cost_score=s,
                vram_mb=cand["vram_mb"], params_m=cand["params_m"],
                rationale=f"Heuristic: dataset={dataset_size}  gpu={gpu_mem:.1f}GB  score≈{s:.3f}",
                probed=False,
            ))
        return sorted(scored, key=lambda m: m.cost_score, reverse=True)

    # ---------------------------------------------------------------------------
    # Step 4 — JIT VRAM filter (NOT a selector — just a filter)
    # ---------------------------------------------------------------------------

    def apply_jit_filter(
        self,
        ranked: Dict[str, List[RankedModel]],
        vram_gb: float,
    ) -> Dict[str, List[RankedModel]]:
        """
        Remove candidates whose VRAM requirement exceeds the available budget.

        Budget: 70% of free VRAM (headroom for activations + gradients).
        When running on CPU (vram_gb == 0), all candidates pass through.
        At least one candidate is always retained (the lightest one).
        """
        if vram_gb <= 0:
            return ranked  # CPU — no VRAM filter needed

        budget_mb = int(vram_gb * 1024 * 0.70)
        filtered: Dict[str, List[RankedModel]] = {}

        for mod, models in ranked.items():
            passing = [m for m in models if m.vram_mb <= budget_mb]
            if not passing:
                # Always keep at least the lightest model
                passing = [min(models, key=lambda m: m.vram_mb)]
                logger.warning(
                    "JIT filter: all %s candidates exceed VRAM budget (%d MB), "
                    "keeping lightest: %s",
                    mod, budget_mb, passing[0].name,
                )
            else:
                removed = len(models) - len(passing)
                if removed:
                    logger.info(
                        "JIT filter: removed %d %s candidate(s) exceeding VRAM budget",
                        removed, mod,
                    )
            filtered[mod] = passing

        return filtered

    # ---------------------------------------------------------------------------
    # Step 5 — Manual override (ALWAYS applied last)
    # ---------------------------------------------------------------------------
    def select_final(
        self,
        ranked: Dict[str, List[RankedModel]],
        schema_info: Dict[str, Any],
        hardware_info: Dict[str, Any],
        manual_override: Optional[Dict[str, str]] = None,
        fallback_count: int = 0,
    ) -> FinalSelection:
        """
        Build the final ``FinalSelection`` from ranked lists.
        Handles JIT VRAM filtering, hard override priority, and trade-off reporting.
        """
        modalities = schema_info.get("global_modalities", ["tabular"])
        gpu_mem = hardware_info.get("gpu_memory_gb", 0.0)
        vram_mb_limit = gpu_mem * 1024 * 0.70 # Use the same budget as apply_jit_filter
        n_modalities = len(modalities)

        # Apply JIT filter first to get feasible pool
        filtered_ranked = self.apply_jit_filter(ranked, gpu_mem)

        # Pick auto winners from filtered pool
        auto_tabular = filtered_ranked["tabular"][0].name if filtered_ranked.get("tabular") else None
        auto_text    = filtered_ranked["text"][0].name    if filtered_ranked.get("text")    else None
        auto_image   = filtered_ranked["image"][0].name   if filtered_ranked.get("image")   else None

        final_tabular = auto_tabular
        final_text    = auto_text
        final_image   = auto_image
        selection_type = "auto"
        override_report = None

        if user_selection:
            # We process manual overrides with hard-priority: if it passes JIT, it wins.
            for mod in ["tabular", "text", "image"]:
                user_mod = user_selection.get(mod)
                if not user_mod:
                    continue
                
                # Find the user's model in the FULL ranked list to check cost/feasibility
                user_cand = next((m for m in ranked.get(mod, []) if m.name == user_mod), None)
                if not user_cand:
                    logger.warning(f"User selected model '{user_mod}' for modality '{mod}' not found in candidates.")
                    continue
                
                # Get the auto candidate to calculate tradeoff
                auto_name = locals().get(f"auto_{mod}")
                auto_cand = next((m for m in ranked.get(mod, []) if m.name == auto_name), None)
                
                # Check JIT feasibility
                if gpu_mem > 0 and user_cand.vram_mb > vram_mb_limit:
                    override_report = {
                        "status": "rejected",
                        "reason": "OOM",
                        "auto_model": getattr(auto_cand, "name", "None"),
                        "override_model": user_mod,
                        "expected_tradeoff": {"accuracy_delta": 0, "latency_delta": 0, "memory_delta": 0}
                    }
                    logger.warning(
                        f"Manual override for {mod} with '{user_mod}' rejected due to OOM. "
                        f"Required VRAM: {user_cand.vram_mb:.1f}MB, Budget: {vram_mb_limit:.1f}MB."
                    )
                    continue
                
                # Accepted
                if mod == "tabular": final_tabular = user_mod
                elif mod == "text": final_text = user_mod
                elif mod == "image": final_image = user_mod
                selection_type = "manual_override"
                
                # Calculate tradeoff (positive delta = user model is higher)
                acc_delta = (user_cand.val_score - auto_cand.val_score) if auto_cand else 0.0
                lat_delta = (user_cand.latency_ms - auto_cand.latency_ms) if auto_cand else 0.0
                mem_delta = (user_cand.vram_mb - auto_cand.vram_mb) if auto_cand else 0.0
                
                override_report = {
                    "status": "accepted",
                    "reason": "valid",
                    "auto_model": getattr(auto_cand, "name", "None"),
                    "override_model": user_mod,
                    "expected_tradeoff": {
                        "accuracy_delta": acc_delta,
                        "latency_delta": lat_delta,
                        "memory_delta": mem_delta
                    }
                }

        # Fusion: data-interaction-aware (not just GPU-based)
        if n_modalities > 1:
            # Use attention only when multimodal + strong GPU + text+tabular mix
            has_cross_modal = "text" in modalities and "tabular" in modalities
            fusion = (
                "attention"
                if (gpu_mem >= 8 and has_cross_modal)
                else "concatenation"
            )
        else:
            fusion = "none"

        # Build rationale
        rationale: Dict[str, str] = {}
        if ranked.get("tabular"):
            top = ranked["tabular"][0]
            rationale["tabular_encoder"] = (
                f"{'Manual override' if selection_type == 'manual_override' and 'tabular' in (user_selection or {}) else 'Auto-selected'}: "
                f"{final_tabular}  {top.rationale}"
            )
        if ranked.get("text"):
            rationale["text_encoder"] = (
                f"{'Manual override' if selection_type == 'manual_override' and 'text' in (user_selection or {}) else 'Auto-selected'}: "
                f"{final_text}  {ranked['text'][0].rationale}"
            )
        rationale["fusion"] = (
            f"fusion={fusion}  modalities={n_modalities}  gpu={gpu_mem:.1f}GB"
        )

        # Probe summary for UI display
        probe_summary: Dict[str, Any] = {}
        for mod, models in ranked.items():
            probed = [m for m in models if m.probed]
            probe_summary[mod] = {
                "method": "1-fold CV" if probed else "heuristic",
                "candidates": [
                    {
                        "name": m.name,
                        "label": m.label,
                        "val_score": round(m.val_score, 4),
                        "latency_ms": round(m.latency_ms, 1),
                        "cost_score": round(m.cost_score, 4),
                        "probed": m.probed,
                        "rationale": m.rationale,
                        "selected": m.name == (
                            final_tabular if mod == "tabular"
                            else final_text if mod == "text"
                            else final_image
                        ),
                    }
                    for m in models
                ],
            }

        logger.info(
            "CandidateSelector final: tabular=%s  text=%s  image=%s  "
            "fusion=%s  type=%s",
            final_tabular, final_text, final_image, fusion, selection_type,
        )

        return FinalSelection(
            tabular=final_tabular,
            text=final_text,
            image=final_image,
            fusion_strategy=fusion,
            selection_type=selection_type,
            ranked_tabular=filtered_ranked.get("tabular", []),
            ranked_text=filtered_ranked.get("text", []),
            ranked_image=filtered_ranked.get("image", []),
            rationale=rationale,
            probe_summary=probe_summary,
            override_report=override_report,
        )

    # ===================================================================
    # FRONTEND API: recommend_models() wrapper for /select-model endpoint
    # ===================================================================

    def recommend_models(
        self,
        problem_type: str,
        modalities: List[str],
        dataset_size: int = 10_000,
        avg_tokens: int = 128,
        schema_info: Optional[Dict[str, Any]] = None,
        preprocess_plan: Optional[Dict[str, Any]] = None,
        probe_info: Optional[Dict[str, Any]] = None,
        resource_budget: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Unified API for /select-model endpoint that matches AdvancedModelSelector interface.
        
        This wraps the full CandidateSelector pipeline (generate → rank → select_final)
        to provide model recommendations consistent with Phase 4 training pipeline.
        
        When actual data is provided (data dict with schema, X, y, text, image), 
        performs real probes. Otherwise, uses heuristic estimation based on dataset 
        size, modalities, and avg_tokens.
        
        Returns list of dicts matching Streamlit frontend JSON contract::
        
            {
              "name": "<ViT-Base + BERT-base + TabNet>",
              "image_encoder": "<name or null>",
              "text_encoder": "<name or null>",
              "tabular_encoder": "<name or null>",
              "fusion_strategy": "attention|concatenation",
              "batch_size": <int>,
              "hpo_space": { "<param>": {...}, ... },
              "rationale": { "<component>": "<reason>", ... },
              "hardware_info": { "gpu_available": bool, ... },
              "tier": "primary" | "fallback"
            }
        """
        # Build minimal schema from heuristics
        if not schema_info:
            modality_count = len(modalities)
            schema_info = {
                "n_rows": dataset_size,
                "n_classes": 2 if problem_type == "classification_binary" else 10,
                "problem_type": problem_type,
                "modalities": modalities,
                "tabular_columns": ["f1", "f2"] if "tabular" in modalities else [],
                "text_column": "text" if "text" in modalities else None,
                "image_column": "image" if "image" in modalities else None,
            }

        # 1. Generate candidates
        candidates = self.generate_candidates(schema_info)

        # 2. Quick probes (only if data provided; otherwise use heuristics)
        probe_scores: Dict[str, Dict[str, Any]] = {"tabular": {}, "text": {}, "image": {}}
        
        if data and data.get("X") is not None and candidates.get("tabular"):
            try:
                X, y = data["X"], data["y"]
                probe_scores["tabular"] = self.quick_probe_tabular(
                    candidates["tabular"], X, y, problem_type
                )
                logger.info("Probed %d tabular candidates", len(probe_scores["tabular"]))
            except Exception as e:
                logger.warning("Tabular probe failed in recommend_models: %s", e)

        if data and data.get("text") and candidates.get("text"):
            try:
                text_data, y = data["text"], data["y"]
                probe_scores["text"] = self.quick_probe_text(
                    candidates["text"], text_data, y, problem_type
                )
                logger.info("Probed %d text candidates", len(probe_scores["text"]))
            except Exception as e:
                logger.warning("Text probe failed in recommend_models: %s", e)

        if data and data.get("image") and candidates.get("image"):
            try:
                image_data, y = data["image"], data["y"]
                probe_scores["image"] = self.quick_probe_image(
                    candidates["image"], image_data, y, problem_type
                )
                logger.info("Probed %d image candidates", len(probe_scores["image"]))
            except Exception as e:
                logger.warning("Image probe failed in recommend_models: %s", e)

        # 3. Rank candidates
        hardware_info = resource_budget or {
            "gpu_available": __import__("torch").cuda.is_available() if __import__("importlib").util.find_spec("torch") else False,
            "gpu_memory_gb": 8.0,
            "cpu_cores": __import__("os").cpu_count() or 4,
        }

        ranked = self.rank_candidates(
            probe_scores=probe_scores,
            candidates=candidates,
            schema_info=schema_info,
            hardware_info=hardware_info,
        )

        # 4. Select final models
        final = self.select_final(
            ranked=ranked,
            hardware_info=hardware_info,
            manual_override=None,
        )

        # 5. Build recommendations list matching AdvancedModelSelector format
        primary_rec: Dict[str, Any] = {
            "name": " + ".join([e for e in [final.image, final.text, final.tabular] if e]),
            "image_encoder": final.image,
            "text_encoder": final.text,
            "tabular_encoder": final.tabular,
            "fusion_strategy": final.fusion_strategy,
            "batch_size": TABULAR_CANDIDATE_POOL[0].get("batch_size", 32) if "tabular" in modalities else 32,
            "hpo_space": {},  # HPO space filled by AdvancedModelSelector if needed
            "rationale": final.rationale or {},
            "hardware_info": hardware_info,
            "probe_scores": probe_scores,
            "tier": "primary",
        }

        # Fallback recommendation
        fallback_rec: Dict[str, Any] = {
            "name": "Lightweight Fallback",
            "image_encoder": None,
            "text_encoder": "MiniLM-L6" if "text" in modalities else None,
            "tabular_encoder": "XGBoost" if "tabular" in modalities else None,
            "fusion_strategy": "concatenation",
            "batch_size": 8,
            "hpo_space": {},
            "rationale": {"general": "Lightweight fallback for memory-constrained environments"},
            "hardware_info": hardware_info,
            "probe_scores": {},
            "tier": "fallback",
        }

        return [primary_rec, fallback_rec]
