import argparse
import sys

from . import transformers_forecasting as tf


def main(argv):
    parser = argparse.ArgumentParser(description="Train population transformer model")
    args = parser.parse_args(argv)
    print("✅ Training population transformer model...")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
