"""Tests for the PTB-XL inference verification module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from inference.ptb_xl_inference import (
    PTBXLInferenceVerifier,
    _build_ptbxl_metadata,
    _build_ptbxl_ecg_image,
    build_trial_inputs,
    _TabularStubPredictor,
    InferenceVerificationResult,
)
from data_ingestion.adapters.ecg_adapter import ECGAdapter
from data_ingestion.schema_detector import MultiDatasetSchemaDetector


# ---------------------------------------------------------------------------
# Synthetic dataset construction tests
# ---------------------------------------------------------------------------


class TestSyntheticDatasets:
    """Verify that synthetic datasets are well-formed."""

    def test_metadata_shape(self):
        df = _build_ptbxl_metadata(100)
        assert len(df) == 100
        assert "ecg_id" in df.columns
        assert "scp_codes" in df.columns
        assert "diagnostic_superclass" in df.columns

    def test_metadata_columns_present(self):
        df = _build_ptbxl_metadata(50)
        expected = {
            "ecg_id", "patient_id", "age", "sex", "height", "weight",
            "scp_codes", "report", "diagnostic_superclass",
            "filename_lr", "filename_hr", "strat_fold", "recording_date",
        }
        assert expected.issubset(set(df.columns))

    def test_image_shape(self):
        df = _build_ptbxl_ecg_image(100)
        assert len(df) == 100
        assert "image_path" in df.columns
        assert "ecg_id" in df.columns

    def test_image_paths_end_with_png(self):
        df = _build_ptbxl_ecg_image(20)
        assert all(str(p).endswith(".png") for p in df["image_path"])

    def test_datasets_share_ecg_id(self):
        df_meta = _build_ptbxl_metadata(50)
        df_img = _build_ptbxl_ecg_image(50)
        shared = set(df_meta["ecg_id"]) & set(df_img["ecg_id"])
        assert len(shared) == 50


# ---------------------------------------------------------------------------
# ECGAdapter detection tests
# ---------------------------------------------------------------------------


class TestECGAdapterDetection:
    """Verify ECGAdapter correctly identifies synthetic datasets."""

    def setup_method(self):
        self.adapter = ECGAdapter()

    def test_metadata_is_ecg(self):
        df = _build_ptbxl_metadata(50)
        assert self.adapter.is_ecg_dataset(df) is True

    def test_image_is_ecg(self):
        df = _build_ptbxl_ecg_image(50)
        assert self.adapter.is_ecg_dataset(df) is True

    def test_image_dataset_detected(self):
        df = _build_ptbxl_ecg_image(50)
        assert self.adapter.is_ecg_image_dataset(df) is True

    def test_metadata_target_inferred(self):
        df = _build_ptbxl_metadata(50)
        target = self.adapter.infer_ecg_target(df)
        assert target == "diagnostic_superclass"

    def test_image_target_inferred(self):
        df = _build_ptbxl_ecg_image(50)
        target = self.adapter.infer_ecg_target(df)
        assert target == "diagnostic_superclass"

    def test_scp_codes_expansion(self):
        df = _build_ptbxl_metadata(20)
        expanded = self.adapter.expand_scp_codes(df.copy())
        assert "scp_codes_len" in expanded.columns
        assert "scp_primary_code" in expanded.columns

    def test_summarize_metadata(self):
        df = _build_ptbxl_metadata(50)
        summary = self.adapter.summarize(df)
        assert summary["is_ecg"] is True
        assert summary["total_records"] == 50


# ---------------------------------------------------------------------------
# Schema detection tests
# ---------------------------------------------------------------------------


class TestSchemaDetection:
    """Verify Tier-1 and Tier-2 schema detection on synthetic data."""

    def setup_method(self):
        self.detector = MultiDatasetSchemaDetector()

    def test_metadata_schema_valid(self):
        df = _build_ptbxl_metadata(100)
        schema = self.detector.detect_global_schema({"ptbxl_meta": df})
        assert schema.global_problem_type != "unsupervised"
        assert schema.primary_target != "Unknown"
        assert len(schema.global_modalities) >= 1

    def test_image_schema_valid(self):
        df = _build_ptbxl_ecg_image(100)
        schema = self.detector.detect_global_schema({"ptbxl_img": df})
        assert schema.global_problem_type != "unsupervised"

    def test_combined_schema_has_relatedness(self):
        df_meta = _build_ptbxl_metadata(100)
        df_img = _build_ptbxl_ecg_image(100)
        schema = self.detector.detect_global_schema({
            "ptbxl_meta": df_meta,
            "ptbxl_img": df_img,
        })
        report = schema.relatedness_report
        assert "pairwise_scores" in report
        assert "n_groups" in report


# ---------------------------------------------------------------------------
# Combinability tests
# ---------------------------------------------------------------------------


class TestCombinability:
    """Verify the relatedness / combinability check."""

    def setup_method(self):
        self.verifier = PTBXLInferenceVerifier(n_samples=100)

    def test_combinability_returns_score(self):
        df_meta = _build_ptbxl_metadata(100)
        df_img = _build_ptbxl_ecg_image(100)
        result = self.verifier.check_combinability(df_meta, df_img)
        assert "relatedness_score" in result
        assert isinstance(result["relatedness_score"], float)

    def test_combinability_returns_boolean(self):
        df_meta = _build_ptbxl_metadata(100)
        df_img = _build_ptbxl_ecg_image(100)
        result = self.verifier.check_combinability(df_meta, df_img)
        assert isinstance(result["datasets_combinable"], bool)

    def test_unrelated_datasets_not_combined(self):
        """Two completely unrelated DataFrames should not be combined."""
        df_unrelated = pd.DataFrame({
            "x": range(50),
            "y": range(50),
            "label": ["a", "b"] * 25,
        })
        df_ecg = _build_ptbxl_metadata(50)
        result = self.verifier.check_combinability(df_ecg, df_unrelated)
        # The relatedness score should be low
        assert result["relatedness_score"] < 0.5


# ---------------------------------------------------------------------------
# Trial prediction tests
# ---------------------------------------------------------------------------


class TestTrialPrediction:
    """Verify trial input construction and prediction."""

    def test_build_trial_inputs(self):
        inputs = build_trial_inputs(5)
        assert len(inputs) == 5
        assert all("age" in inp for inp in inputs)
        assert all("sex" in inp for inp in inputs)
        assert all("diagnostic_superclass" in inp for inp in inputs)

    def test_stub_predictor_output_shape(self):
        import torch
        predictor = _TabularStubPredictor(input_dim=4, num_classes=5)
        batch = {"tabular": torch.randn(3, 4)}
        out = predictor(batch)
        assert out.shape == (3, 5)

    def test_trial_prediction_produces_results(self):
        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=3)
        df = _build_ptbxl_metadata(50)
        inputs = build_trial_inputs(3)
        result = verifier.run_trial_prediction(df, inputs)
        assert result["n_samples"] == 3
        assert len(result["predictions"]) == 3
        assert len(result["confidences"]) == 3
        assert all(0 < c <= 1.0 for c in result["confidences"])

    def test_prediction_labels_are_valid(self):
        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=5)
        df = _build_ptbxl_metadata(50)
        inputs = build_trial_inputs(5)
        result = verifier.run_trial_prediction(df, inputs)
        valid_classes = set(result["class_labels"])
        assert all(p in valid_classes for p in result["predictions"])


# ---------------------------------------------------------------------------
# Full run integration test
# ---------------------------------------------------------------------------


class TestFullRun:
    """Integration test for the complete verification workflow."""

    def test_full_run_returns_result(self):
        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=3)
        result = verifier.run()
        assert isinstance(result, InferenceVerificationResult)

    def test_full_run_has_verdict(self):
        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=3)
        result = verifier.run()
        assert len(result.efficiency_verdict) > 0
        assert "checks passed" in result.efficiency_verdict

    def test_full_run_individual_analyses(self):
        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=3)
        result = verifier.run()
        assert result.ptbxl_metadata_summary.get("is_ecg_dataset") is True
        assert result.ptbxl_image_summary.get("is_ecg_dataset") is True

    def test_full_run_schemas_detected(self):
        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=3)
        result = verifier.run()
        assert result.ptbxl_metadata_schema.get("primary_target") != "Unknown"
        assert result.ptbxl_image_schema.get("primary_target") != "Unknown"

    def test_full_run_trial_predictions(self):
        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=3)
        result = verifier.run()
        assert len(result.trial_predictions.get("predictions", [])) == 3
        assert len(result.trial_inputs) == 3

    def test_full_run_relatedness_populated(self):
        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=3)
        result = verifier.run()
        assert isinstance(result.relatedness_score, float)
        assert isinstance(result.datasets_combinable, bool)
        assert "n_groups" in result.relatedness_report
