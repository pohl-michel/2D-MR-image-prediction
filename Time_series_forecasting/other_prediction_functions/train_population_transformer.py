import argparse
import json
import os
import sys

from . import transformers_forecasting as tf

CONFIG_DIR = "other_prediction_functions"


def main(argv):
    parser = argparse.ArgumentParser(description="Train population transformer model")
    parser.add_argument(
        "--config",
        help="path to JSON config file",
        default="pop_transformer_training_config_test.json",
        nargs=1,
        type=str,
    )
    args = parser.parse_args(argv)
    print("✅ Training population transformer model...")
    print(f"Using config file: {args.config}")

    with open(os.path.join(CONFIG_DIR, args.config), "r") as f:
        config = json.load(f)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
