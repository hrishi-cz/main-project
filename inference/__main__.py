"""
CLI entry point for the PTB-XL inference verification module.

Usage::

    python -m inference

Runs the full verification workflow, prints the analysis report, then
prompts the user to **manually enter** values for each feature column.
The manually entered input is fed through the stub predictor and the
prediction result is displayed.
"""

import logging
import sys

from inference.ptb_xl_inference import (
    PTBXLInferenceVerifier,
    collect_manual_input,
    _build_ptbxl_metadata,
    _build_ptbxl_ecg_image,
)
import pandas as pd


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    verifier = PTBXLInferenceVerifier(n_samples=200, n_trial=5)
    result = verifier.run()
    result.print_report()

    # Exit with non-zero code if not all checks passed
    if result.checks_passed < result.total_checks:
        sys.exit(1)

    # ------------------------------------------------------------------
    # Interactive manual input for prediction
    # ------------------------------------------------------------------
    feature_cols = result.trial_predictions.get("feature_columns", [])
    class_labels = result.trial_predictions.get("class_labels", [])

    print()
    print("=" * 72)
    print("  INTERACTIVE PREDICTION")
    print("=" * 72)
    print()
    print("  The verification is complete. You can now enter your own")
    print("  feature values to get a prediction.")
    print()
    if class_labels:
        print(f"  Target classes : {class_labels}")
    print(f"  Feature columns: {feature_cols}")
    print()

    while True:
        manual_input = collect_manual_input(feature_cols)

        # Build the combined dataset for prediction context
        df_metadata = _build_ptbxl_metadata(verifier.n_samples)
        df_image = _build_ptbxl_ecg_image(verifier.n_samples)
        if result.datasets_combinable:
            df_combined = pd.merge(
                df_metadata, df_image,
                on="ecg_id", how="inner", suffixes=("", "_img"),
            )
        else:
            df_combined = df_metadata

        pred_result = verifier.run_trial_prediction(df_combined, [manual_input])

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
                    print(f"    {lbl:<8} {p:.4f}  {bar}")
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
