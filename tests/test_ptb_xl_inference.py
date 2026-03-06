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
    collect_manual_input,
    detect_relevant_features,
    _TabularStubPredictor,
    InferenceVerificationResult,
    FEATURE_COLUMNS,
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
        assert all(0 <= c <= 1.0 for c in result["confidences"])

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


# ---------------------------------------------------------------------------
# Print report tests
# ---------------------------------------------------------------------------


class TestPrintReport:
    """Verify the human-readable report output."""

    def test_print_report_returns_string(self):
        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=3)
        result = verifier.run()
        report = result.print_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_print_report_contains_sections(self):
        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=3)
        result = verifier.run()
        report = result.print_report()
        assert "INDIVIDUAL DATASET ANALYSIS" in report
        assert "SCHEMA DETECTION" in report
        assert "DATASET COMBINABILITY" in report
        assert "TRIAL INPUTS" in report
        assert "TRIAL PREDICTIONS" in report
        assert "EFFICIENCY VERDICT" in report

    def test_print_report_contains_trial_data(self):
        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=3)
        result = verifier.run()
        report = result.print_report()
        # Trial inputs should show feature names
        assert "age" in report
        assert "sex" in report
        assert "height" in report
        assert "weight" in report
        # Predictions should show class labels
        for label in result.trial_predictions.get("class_labels", []):
            assert label in report

    def test_print_report_contains_verdict(self):
        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=3)
        result = verifier.run()
        report = result.print_report()
        assert "checks passed" in report
        assert "Relatedness score" in report


# ---------------------------------------------------------------------------
# Manual input collection tests
# ---------------------------------------------------------------------------


class TestManualInput:
    """Verify interactive input collection via collect_manual_input."""

    def test_collects_all_features(self):
        """Simulated user types valid values for every feature."""
        responses = iter(["55", "1", "175.5", "80.0"])
        result = collect_manual_input(_input_fn=lambda _prompt: next(responses))
        assert result == {"age": 55, "sex": 1, "height": 175.5, "weight": 80.0}

    def test_custom_feature_cols(self):
        """Only the requested columns are prompted."""
        responses = iter(["30", "0"])
        result = collect_manual_input(
            ["age", "sex"], _input_fn=lambda _prompt: next(responses),
        )
        assert list(result.keys()) == ["age", "sex"]
        assert result["age"] == 30
        assert result["sex"] == 0

    def test_retries_on_empty_input(self):
        """Empty input is rejected and the prompt repeats."""
        call_count = [0]

        def _fake_input(_prompt):
            call_count[0] += 1
            # First call returns empty, second returns a valid value
            if call_count[0] == 1:
                return ""
            return "42"

        result = collect_manual_input(["age"], _input_fn=_fake_input)
        assert result["age"] == 42
        assert call_count[0] == 2

    def test_retries_on_non_numeric_input(self):
        """Non-numeric input is rejected and the prompt repeats."""
        responses = iter(["abc", "65"])
        result = collect_manual_input(
            ["age"], _input_fn=lambda _prompt: next(responses),
        )
        assert result["age"] == 65

    def test_integer_columns_stay_int(self):
        """Age and sex should be returned as int, not float."""
        responses = iter(["45", "0", "170", "75"])
        result = collect_manual_input(_input_fn=lambda _prompt: next(responses))
        assert isinstance(result["age"], int)
        assert isinstance(result["sex"], int)

    def test_float_columns_stay_float(self):
        """Height and weight with decimals should be float."""
        responses = iter(["45", "1", "170.5", "82.3"])
        result = collect_manual_input(_input_fn=lambda _prompt: next(responses))
        assert isinstance(result["height"], float)
        assert isinstance(result["weight"], float)

    def test_manual_input_runs_prediction(self):
        """Manual input fed to run_trial_prediction produces valid output."""
        responses = iter(["55", "1", "175.5", "80.0"])
        manual = collect_manual_input(_input_fn=lambda _prompt: next(responses))

        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=1)
        df = _build_ptbxl_metadata(50)
        pred = verifier.run_trial_prediction(df, [manual])
        assert pred["n_samples"] == 1
        assert len(pred["predictions"]) == 1
        assert 0 <= pred["confidences"][0] <= 1.0

    def test_categorical_input_accepted(self):
        """Categorical columns accept string values."""
        responses = iter(["55", "NORM"])
        result = collect_manual_input(
            ["age", "diag"],
            column_types={"age": "numeric", "diag": "categorical"},
            _input_fn=lambda _prompt: next(responses),
        )
        assert result["age"] == 55
        assert result["diag"] == "NORM"

    def test_categorical_retries_on_empty(self):
        """Categorical columns still reject empty input."""
        responses = iter(["", "MI"])
        result = collect_manual_input(
            ["diag"],
            column_types={"diag": "categorical"},
            _input_fn=lambda _prompt: next(responses),
        )
        assert result["diag"] == "MI"


# ---------------------------------------------------------------------------
# Relevant feature detection tests
# ---------------------------------------------------------------------------


class TestDetectRelevantFeatures:
    """Verify auto-detection of relevant feature columns from a dataset."""

    def test_filters_out_ids_from_metadata(self):
        """patient_id and ecg_id should be filtered out as ID columns."""
        df = _build_ptbxl_metadata(100)
        info = detect_relevant_features(df)
        kept = info["feature_columns"]
        dropped = info["dropped_columns"]
        # ID-like columns should be dropped
        assert "patient_id" not in kept
        assert "ecg_id" not in kept
        # Real features should be kept
        assert "age" in kept
        assert "sex" in kept

    def test_filters_out_paths(self):
        """File-path columns like filename_lr should be dropped."""
        df = _build_ptbxl_metadata(100)
        info = detect_relevant_features(df)
        kept = info["feature_columns"]
        assert "filename_lr" not in kept
        assert "filename_hr" not in kept

    def test_detects_target_column(self):
        """Target column should be detected and excluded from features."""
        df = _build_ptbxl_metadata(100)
        info = detect_relevant_features(df)
        target = info["target_column"]
        assert target == "diagnostic_superclass"
        assert target not in info["feature_columns"]

    def test_column_types_populated(self):
        """column_types dict should map each kept column to numeric or categorical."""
        df = _build_ptbxl_metadata(100)
        info = detect_relevant_features(df)
        for col in info["feature_columns"]:
            assert col in info["column_types"]
            assert info["column_types"][col] in ("numeric", "categorical")

    def test_custom_target_column(self):
        """Explicit target_col argument is honoured."""
        df = pd.DataFrame({
            "a": range(100),
            "b": np.random.randn(100),
            "target": ["x", "y"] * 50,
        })
        # Force string cols to object dtype for compatibility
        for c in df.select_dtypes(include=["string", "str"]).columns:
            df[c] = df[c].astype(object)
        info = detect_relevant_features(df, target_col="target")
        assert info["target_column"] == "target"
        assert "target" not in info["feature_columns"]

    def test_dynamic_features_run_prediction(self):
        """Detected features can drive run_trial_prediction."""
        df = _build_ptbxl_metadata(100)
        info = detect_relevant_features(df)
        # Only use the numeric features for the stub predictor
        numeric_cols = [c for c in info["feature_columns"]
                        if info["column_types"].get(c) == "numeric"]
        # Build a manual input from the detected columns
        manual = {c: 0.0 for c in numeric_cols}
        manual["age"] = 55.0
        manual["sex"] = 1.0

        verifier = PTBXLInferenceVerifier(n_samples=50, n_trial=1)
        pred = verifier.run_trial_prediction(
            df, [manual], feature_cols=numeric_cols,
        )
        assert pred["n_samples"] == 1
        assert len(pred["predictions"]) == 1
