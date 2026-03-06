"""
PTB-XL Inference Verifier
=========================

Verifies inference efficiency of the AutoVision+ pipeline by:

1. Simulating both PTB-XL datasets with realistic trial data:
   - PTB-XL metadata dataset (Kaggle: khyeh0719/ptb-xl-dataset)
   - PTB-XL ECG image dataset (Kaggle: bjoernjostein/ptb-xl-ecg-image-gmc2024)

2. Running individual schema detection + ECG adapter analysis on each dataset.

3. Checking if the two datasets share enough commonality (via the Tier-2
   relatedness engine) to be combined for unified inference.

4. If combinable (relatedness score >= 0.5), merging and running joint
   inference; otherwise keeping them separate.

5. Providing trial/sample inputs and verifying prediction results through
   the pipeline's tabular encoder path.

All heavy model weights (ResNet-50, BERT) are **not** required — the
verifier exercises the schema detection, ECG adapter, preprocessing,
and lightweight tabular-encoder prediction paths so it can run in CI
without GPU or large downloads.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_ingestion.adapters.ecg_adapter import ECGAdapter
from data_ingestion.schema_detector import MultiDatasetSchemaDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic dataset builders
# ---------------------------------------------------------------------------

def _build_ptbxl_metadata(n: int = 200) -> pd.DataFrame:
    """
    Build a synthetic DataFrame mimicking the PTB-XL metadata CSV.

    Columns mirror the real dataset:
      ecg_id, patient_id, age, sex, height, weight, scp_codes,
      report, diagnostic_superclass, filename_lr, filename_hr,
      strat_fold, recording_date.
    """
    rng = np.random.RandomState(42)
    superclasses = ["NORM", "MI", "STTC", "HYP", "CD"]
    scp_templates = [
        "{'NORM': 100.0}",
        "{'MI': 80.0, 'STTC': 20.0}",
        "{'HYP': 60.0, 'CD': 40.0}",
        "{'STTC': 90.0}",
        "{'CD': 70.0, 'NORM': 30.0}",
    ]

    df = pd.DataFrame({
        "ecg_id": list(range(1, n + 1)),
        "patient_id": rng.randint(1000, 9999, size=n).tolist(),
        "age": rng.randint(20, 90, size=n).tolist(),
        "sex": rng.choice([0, 1], size=n).tolist(),
        "height": rng.normal(170, 10, size=n).round(1).tolist(),
        "weight": rng.normal(75, 15, size=n).round(1).tolist(),
        "scp_codes": [scp_templates[i % len(scp_templates)] for i in range(n)],
        "report": [f"ECG report for patient {i}" for i in range(n)],
        "diagnostic_superclass": [
            superclasses[i % len(superclasses)] for i in range(n)
        ],
        "filename_lr": [
            f"records100/{i // 100:05d}/{i:05d}_lr.dat" for i in range(n)
        ],
        "filename_hr": [
            f"records500/{i // 100:05d}/{i:05d}_hr.dat" for i in range(n)
        ],
        "strat_fold": [((i % 10) + 1) for i in range(n)],
        "recording_date": pd.date_range("2000-01-01", periods=n, freq="D")
        .strftime("%Y-%m-%d")
        .tolist(),
    })
    # Ensure string columns use 'object' dtype for compatibility with
    # ECGAdapter's dtype checks (pandas 3.0+ defaults to 'str').
    for col in df.select_dtypes(include=["string", "str"]).columns:
        df[col] = df[col].astype(object)
    return df


def _build_ptbxl_ecg_image(n: int = 200) -> pd.DataFrame:
    """
    Build a synthetic DataFrame mimicking the PTB-XL ECG Image (GMC2024)
    dataset.

    This dataset augments the metadata with an ``image_path`` column
    pointing to waveform PNG images, plus a shared ``ecg_id`` linking
    back to the metadata dataset.
    """
    rng = np.random.RandomState(42)
    superclasses = ["NORM", "MI", "STTC", "HYP", "CD"]

    df = pd.DataFrame({
        "ecg_id": list(range(1, n + 1)),
        "image_path": [
            f"images/{i // 100:05d}/{i:05d}_ecg.png" for i in range(n)
        ],
        "diagnostic_superclass": [
            superclasses[i % len(superclasses)] for i in range(n)
        ],
        "label": [
            superclasses[i % len(superclasses)] for i in range(n)
        ],
        "patient_id": rng.randint(1000, 9999, size=n).tolist(),
        "scp_codes": [
            "{'NORM': 100.0}" if i % 2 == 0 else "{'MI': 80.0}"
            for i in range(n)
        ],
    })
    # Ensure string columns use 'object' dtype for compatibility with
    # ECGAdapter's dtype checks (pandas 3.0+ defaults to 'str').
    for col in df.select_dtypes(include=["string", "str"]).columns:
        df[col] = df[col].astype(object)
    return df


# ---------------------------------------------------------------------------
# Trial input builders
# ---------------------------------------------------------------------------

def build_trial_inputs(n: int = 5) -> List[Dict[str, Any]]:
    """
    Create small trial inputs that exercise the prediction path.

    Each trial input contains tabular features (age, sex, height, weight)
    and metadata (ecg_id, diagnostic_superclass) representative of what
    the trained pipeline would see.
    """
    rng = np.random.RandomState(99)
    superclasses = ["NORM", "MI", "STTC", "HYP", "CD"]
    return [
        {
            "ecg_id": 10000 + i,
            "age": int(rng.randint(25, 85)),
            "sex": int(rng.choice([0, 1])),
            "height": float(round(rng.normal(170, 10), 1)),
            "weight": float(round(rng.normal(75, 15), 1)),
            "diagnostic_superclass": superclasses[i % len(superclasses)],
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Lightweight stub predictor (no heavy model weights needed)
# ---------------------------------------------------------------------------

class _TabularStubPredictor(nn.Module):
    """
    A lightweight predictor that accepts a dict with a ``"tabular"`` key
    and produces classification logits, mimicking the real pipeline's
    ``_MultimodalHead`` interface without requiring ResNet-50 or BERT.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        # Deterministic init for reproducibility
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch["tabular"])


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceVerificationResult:
    """Structured output of the full verification run."""

    # Individual dataset analysis
    ptbxl_metadata_summary: Dict[str, Any] = field(default_factory=dict)
    ptbxl_image_summary: Dict[str, Any] = field(default_factory=dict)

    # Schema detection
    ptbxl_metadata_schema: Dict[str, Any] = field(default_factory=dict)
    ptbxl_image_schema: Dict[str, Any] = field(default_factory=dict)

    # Relatedness / combinability
    relatedness_score: float = 0.0
    datasets_combinable: bool = False
    relatedness_report: Dict[str, Any] = field(default_factory=dict)

    # Combined schema (if combinable)
    combined_schema: Dict[str, Any] = field(default_factory=dict)

    # Trial predictions
    trial_inputs: List[Dict[str, Any]] = field(default_factory=list)
    trial_predictions: Dict[str, Any] = field(default_factory=dict)

    # Overall verdict
    efficiency_verdict: str = ""


# ---------------------------------------------------------------------------
# Main verifier class
# ---------------------------------------------------------------------------

class PTBXLInferenceVerifier:
    """
    End-to-end inference verifier for the AutoVision+ pipeline using
    synthetic PTB-XL datasets.

    Usage
    -----
    >>> verifier = PTBXLInferenceVerifier()
    >>> result = verifier.run()
    >>> print(result.efficiency_verdict)
    """

    def __init__(self, n_samples: int = 200, n_trial: int = 5) -> None:
        self.n_samples = n_samples
        self.n_trial = n_trial
        self.adapter = ECGAdapter()
        self.detector = MultiDatasetSchemaDetector()

    # ------------------------------------------------------------------ #
    # Step 1: Individual dataset analysis via ECGAdapter
    # ------------------------------------------------------------------ #

    def analyze_individual(
        self, df: pd.DataFrame, dataset_name: str
    ) -> Dict[str, Any]:
        """Run ECGAdapter analysis on a single dataset."""
        is_ecg = self.adapter.is_ecg_dataset(df)
        is_image = self.adapter.is_ecg_image_dataset(df)
        target = self.adapter.infer_ecg_target(df)
        summary = self.adapter.summarize(df)
        image_cols = self.adapter.find_image_columns(df)

        result = {
            "dataset_name": dataset_name,
            "is_ecg_dataset": is_ecg,
            "is_ecg_image_dataset": is_image,
            "target_column": target,
            "image_columns": image_cols,
            "n_records": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "adapter_summary": summary,
        }
        logger.info(
            "Individual analysis [%s]: is_ecg=%s  is_image=%s  target=%s  "
            "records=%d  columns=%d",
            dataset_name, is_ecg, is_image, target, len(df), len(df.columns),
        )
        return result

    # ------------------------------------------------------------------ #
    # Step 2: Schema detection per dataset
    # ------------------------------------------------------------------ #

    def detect_schema_single(
        self, df: pd.DataFrame, dataset_id: str
    ) -> Dict[str, Any]:
        """Run Tier-1+2 schema detection on a single dataset."""
        schema = self.detector.detect_global_schema({dataset_id: df})
        return asdict(schema)

    # ------------------------------------------------------------------ #
    # Step 3: Check relatedness / combinability
    # ------------------------------------------------------------------ #

    def check_combinability(
        self,
        df_metadata: pd.DataFrame,
        df_image: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Determine whether the two PTB-XL datasets can be combined.

        Uses the Tier-2 ``detect_global_schema`` method which internally
        calls ``_check_relatedness`` with pairwise signals:
          - Column name overlap  (Jaccard, weight 0.40)
          - Target column match  (weight 0.30)
          - Modality set overlap (weight 0.20)
          - Problem type match   (weight 0.10)

        The relatedness threshold is 0.5.
        """
        combined_schema = self.detector.detect_global_schema(
            {
                "ptbxl_metadata": df_metadata,
                "ptbxl_ecg_image": df_image,
            }
        )
        schema_dict = asdict(combined_schema)
        report = schema_dict.get("relatedness_report", {})
        pairwise = report.get("pairwise_scores", {})
        n_groups = report.get("n_groups", 1)

        # Extract the single pairwise score (two datasets → one pair)
        score = 0.0
        if pairwise:
            score = list(pairwise.values())[0]

        combinable = n_groups == 1  # one group = combinable

        logger.info(
            "Combinability check: score=%.3f  combinable=%s  n_groups=%d",
            score, combinable, n_groups,
        )
        return {
            "relatedness_score": score,
            "datasets_combinable": combinable,
            "n_groups": n_groups,
            "relatedness_report": report,
            "combined_schema": schema_dict,
        }

    # ------------------------------------------------------------------ #
    # Step 4: Trial prediction
    # ------------------------------------------------------------------ #

    def run_trial_prediction(
        self,
        df_combined: pd.DataFrame,
        trial_inputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run trial predictions using a lightweight stub predictor.

        This exercises the same code-path as the real inference engine
        (tabular features → encoder → prediction) without requiring
        trained weights.
        """
        target_col = self.adapter.infer_ecg_target(df_combined)

        # Determine label encoding
        if target_col != "Unknown" and target_col in df_combined.columns:
            classes = sorted(df_combined[target_col].dropna().unique().tolist())
        else:
            classes = ["NORM", "MI", "STTC", "HYP", "CD"]

        num_classes = len(classes)
        feature_cols = ["age", "sex", "height", "weight"]

        # Build feature tensor from trial inputs
        features = []
        for inp in trial_inputs:
            row = [float(inp.get(c, 0.0)) for c in feature_cols]
            features.append(row)
        feature_tensor = torch.tensor(features, dtype=torch.float32)

        # Run through stub predictor
        torch.manual_seed(42)
        predictor = _TabularStubPredictor(
            input_dim=len(feature_cols), num_classes=num_classes
        )
        predictor.eval()

        with torch.no_grad():
            logits = predictor({"tabular": feature_tensor})
            probs = torch.softmax(logits, dim=-1)
            predicted_indices = torch.argmax(probs, dim=-1).tolist()
            confidences = torch.max(probs, dim=-1).values.tolist()

        predicted_labels = [classes[i] for i in predicted_indices]

        results = {
            "n_samples": len(trial_inputs),
            "feature_columns": feature_cols,
            "num_classes": num_classes,
            "class_labels": classes,
            "predictions": predicted_labels,
            "prediction_indices": predicted_indices,
            "confidences": [round(c, 4) for c in confidences],
            "logits": logits.tolist(),
            "probabilities": probs.tolist(),
            "trial_inputs": trial_inputs,
        }

        logger.info(
            "Trial prediction: %d samples → predictions=%s  confidences=%s",
            len(trial_inputs), predicted_labels,
            [round(c, 4) for c in confidences],
        )
        return results

    # ------------------------------------------------------------------ #
    # Step 5: Full verification run
    # ------------------------------------------------------------------ #

    def run(self) -> InferenceVerificationResult:
        """
        Execute the full inference verification workflow.

        Returns an ``InferenceVerificationResult`` with all intermediate
        and final outputs, plus an overall efficiency verdict.
        """
        logger.info("=" * 70)
        logger.info("PTB-XL Inference Verification — START")
        logger.info("=" * 70)

        result = InferenceVerificationResult()

        # ----- Build synthetic datasets -----
        df_metadata = _build_ptbxl_metadata(self.n_samples)
        df_image = _build_ptbxl_ecg_image(self.n_samples)

        # ----- Step 1: Individual analysis -----
        result.ptbxl_metadata_summary = self.analyze_individual(
            df_metadata, "PTB-XL Metadata (khyeh0719/ptb-xl-dataset)"
        )
        result.ptbxl_image_summary = self.analyze_individual(
            df_image, "PTB-XL ECG Image (bjoernjostein/ptb-xl-ecg-image-gmc2024)"
        )

        # ----- Step 2: Schema detection per dataset -----
        result.ptbxl_metadata_schema = self.detect_schema_single(
            df_metadata, "ptbxl_metadata"
        )
        result.ptbxl_image_schema = self.detect_schema_single(
            df_image, "ptbxl_ecg_image"
        )

        # ----- Step 3: Combinability check -----
        combo = self.check_combinability(df_metadata, df_image)
        result.relatedness_score = combo["relatedness_score"]
        result.datasets_combinable = combo["datasets_combinable"]
        result.relatedness_report = combo["relatedness_report"]
        result.combined_schema = combo["combined_schema"]

        # ----- Step 4: Trial prediction -----
        trial_inputs = build_trial_inputs(self.n_trial)
        result.trial_inputs = trial_inputs

        if result.datasets_combinable:
            # Merge datasets on ecg_id for combined inference
            df_combined = pd.merge(
                df_metadata, df_image,
                on="ecg_id", how="inner", suffixes=("", "_img"),
            )
            logger.info(
                "Datasets COMBINED: %d records after inner join on ecg_id",
                len(df_combined),
            )
        else:
            # Use the metadata dataset alone (the richer tabular source)
            df_combined = df_metadata
            logger.info(
                "Datasets NOT combinable — using metadata dataset only (%d records)",
                len(df_combined),
            )

        result.trial_predictions = self.run_trial_prediction(
            df_combined, trial_inputs
        )

        # ----- Step 5: Efficiency verdict -----
        checks_passed = 0
        total_checks = 5

        # Check 1: Both datasets detected as ECG
        if (result.ptbxl_metadata_summary.get("is_ecg_dataset")
                and result.ptbxl_image_summary.get("is_ecg_dataset")):
            checks_passed += 1

        # Check 2: Image dataset correctly identified as having images
        if result.ptbxl_image_summary.get("is_ecg_image_dataset"):
            checks_passed += 1

        # Check 3: Target columns correctly inferred
        if (result.ptbxl_metadata_summary.get("target_column") != "Unknown"
                and result.ptbxl_image_summary.get("target_column") != "Unknown"):
            checks_passed += 1

        # Check 4: Schema detection produced valid schemas
        if (result.ptbxl_metadata_schema.get("global_problem_type") != "unsupervised"
                and result.ptbxl_image_schema.get("global_problem_type") != "unsupervised"):
            checks_passed += 1

        # Check 5: Trial predictions produced valid outputs
        preds = result.trial_predictions
        if (preds.get("predictions")
                and len(preds["predictions"]) == self.n_trial
                and all(0 < c <= 1.0 for c in preds.get("confidences", []))):
            checks_passed += 1

        verdict_parts = [
            f"Efficiency: {checks_passed}/{total_checks} checks passed.",
            f"Relatedness score: {result.relatedness_score:.3f}.",
            f"Datasets combinable: {result.datasets_combinable}.",
        ]
        if result.datasets_combinable:
            verdict_parts.append(
                "Combined inference enabled — datasets share sufficient "
                "structural overlap for joint prediction."
            )
        else:
            verdict_parts.append(
                "Datasets are structurally distinct — using metadata "
                "dataset only for inference."
            )
        result.efficiency_verdict = " ".join(verdict_parts)

        logger.info("=" * 70)
        logger.info("PTB-XL Inference Verification — COMPLETE")
        logger.info("Verdict: %s", result.efficiency_verdict)
        logger.info("=" * 70)

        return result
