"""
CLI entry point for the PTB-XL inference verification module.

Usage::

    python -m inference

Runs the full verification workflow and prints all trial inputs,
prediction results, dataset analysis, combinability check, and the
overall efficiency verdict to stdout.
"""

import logging
import sys

from inference.ptb_xl_inference import PTBXLInferenceVerifier


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    verifier = PTBXLInferenceVerifier(n_samples=200, n_trial=5)
    result = verifier.run()
    result.print_report()

    # Exit with non-zero code if not all checks passed
    if "5/5" not in result.efficiency_verdict:
        sys.exit(1)


if __name__ == "__main__":
    main()
