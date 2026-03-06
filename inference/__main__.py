"""
CLI entry point for the PTB-XL inference verification module.

Usage::

    python -m inference                   # uses built-in synthetic PTB-XL data
    python -m inference path/to/data.csv  # loads a real CSV dataset

Runs the full verification workflow, prints the analysis report, then
prompts the user to **manually enter** values for each *relevant*
feature column.  Irrelevant columns (IDs, timestamps, file paths, etc.)
are automatically filtered out so the user only fills in what the model
actually needs.
"""

import logging
import sys

from inference.ptb_xl_inference import (
    PTBXLInferenceVerifier,
    collect_manual_input,
    detect_relevant_features,
    _build_ptbxl_metadata,
    _build_ptbxl_ecg_image,
)
import pandas as pd


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ------------------------------------------------------------------
    # Step 1: Load or build the dataset
    # ------------------------------------------------------------------
    csv_path: str | None = sys.argv[1] if len(sys.argv) > 1 else None

    if csv_path:
        print(f"\n  Loading CSV dataset: {csv_path}")
        try:
            df_dataset = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  ✗ Failed to load CSV: {exc}")
            sys.exit(1)
        print(f"  ✓ Loaded {len(df_dataset)} rows × {len(df_dataset.columns)} columns")
        print(f"  Columns: {list(df_dataset.columns)}")
    else:
        df_dataset = None

    # ------------------------------------------------------------------
    # Step 2: Run the verification workflow (always runs on synthetic data)
    # ------------------------------------------------------------------
    verifier = PTBXLInferenceVerifier(n_samples=200, n_trial=5)
    result = verifier.run()
    result.print_report()

    if result.checks_passed < result.total_checks:
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Determine relevant features for prediction input
    # ------------------------------------------------------------------
    if df_dataset is not None:
        # User provided a CSV — detect relevant columns from it
        feat_info = detect_relevant_features(df_dataset)
        feature_cols = feat_info["feature_columns"]
        column_types = feat_info["column_types"]
        dropped_cols = feat_info["dropped_columns"]
        target_col = feat_info["target_column"]

        # Determine class labels from the target column
        if target_col and target_col in df_dataset.columns:
            class_labels = sorted(
                df_dataset[target_col].dropna().unique().tolist()
            )
        else:
            class_labels = result.trial_predictions.get("class_labels", [])
    else:
        # No CSV — use the default features from verification run
        feature_cols = result.trial_predictions.get("feature_columns", [])
        column_types = {}
        dropped_cols = []
        target_col = ""
        class_labels = result.trial_predictions.get("class_labels", [])

    # ------------------------------------------------------------------
    # Step 4: Interactive prediction loop
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("  INTERACTIVE PREDICTION")
    print("=" * 72)
    print()
    print("  The verification is complete. You can now enter your own")
    print("  feature values to get a prediction.")
    print()
    if target_col:
        print(f"  Target column  : {target_col}")
    if class_labels:
        print(f"  Target classes : {class_labels}")
    print(f"  Feature columns: {feature_cols}")
    if dropped_cols:
        print(f"  Auto-filtered  : {dropped_cols}")
    print()

    while True:
        manual_input = collect_manual_input(
            feature_cols,
            column_types=column_types,
        )

        # Build the dataset context for prediction
        if df_dataset is not None:
            df_combined = df_dataset
        else:
            df_metadata = _build_ptbxl_metadata(verifier.n_samples)
            df_image = _build_ptbxl_ecg_image(verifier.n_samples)
            if result.datasets_combinable:
                df_combined = pd.merge(
                    df_metadata, df_image,
                    on="ecg_id", how="inner", suffixes=("", "_img"),
                )
            else:
                df_combined = df_metadata

        pred_result = verifier.run_trial_prediction(
            df_combined, [manual_input], feature_cols=feature_cols,
        )

        # Display the prediction
        sep = "=" * 72
        thin = "-" * 50
        print(sep)
        print("  PREDICTION RESULT")
        print(thin)
        preds = pred_result.get("predictions", [])
        confs = pred_result.get("confidences", [])
        probs = pred_result.get("probabilities", [])
        labels = pred_result.get("class_labels", [])

        _BAR_WIDTH = 30  # max characters for the probability bar

        if preds:
            print(f"  Predicted class : {preds[0]}")
            print(f"  Confidence      : {confs[0]:.4f}")
            print()
            if probs and labels:
                print("  Class probabilities:")
                for lbl, p in zip(labels, probs[0]):
                    bar = "█" * int(p * _BAR_WIDTH)
                    print(f"    {str(lbl):<12} {p:.4f}  {bar}")
        print(sep)
        print()

        try:
            again = input("  Run another prediction? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if again not in ("y", "yes"):
            break

    print()
    print("  Done. Goodbye!")


if __name__ == "__main__":
    main()
