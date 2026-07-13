"""Generate IFCS test-set predictions from a saved final model bundle.

Example:

    python predict_from_saved_models.py --test test_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import challenge_pipeline as cp


def run_inference(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    outdir = Path(args.outdir)
    cp.ensure_dir(outdir)

    trained, metadata = cp.load_trained_models(model_dir)
    threshold_map = metadata["threshold_map"]
    test_df = cp.load_frame(Path(args.test))
    cp.predict_test(trained, test_df, threshold_map, outdir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate test predictions using saved IFCS trained models."
    )
    parser.add_argument(
        "--test",
        default="test_features.csv",
        help="Path to the challenge test feature CSV.",
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/models",
        help="Directory containing saved model_metadata.json and model files.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Directory for test probabilities and candidate prediction CSVs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
