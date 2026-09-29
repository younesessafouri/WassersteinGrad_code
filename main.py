"""Run an experiment of the paper: python main.py configs/xai.yaml [--mode eval]"""

import argparse
import logging

from py4castxai.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="WassersteinGrad experiments on a Py4Cast model.")
    parser.add_argument("config", help="experiment config, see configs/xai.yaml")
    parser.add_argument("--mode", help="overrides the mode set in the config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment(args.config, args.mode)


if __name__ == "__main__":
    main()
