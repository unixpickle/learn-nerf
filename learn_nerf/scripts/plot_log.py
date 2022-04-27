"""
Plot one or more training logs, saved as txt files.
"""

import argparse
import os
from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=str, default="fine")
    parser.add_argument("--log_scale", action="store_true")
    parser.add_argument("log_paths", nargs="+")
    args = parser.parse_args()

    for path in args.log_paths:
        label = label_for_path(path)
        log = read_log(path)
        values = log[args.field]
        plt.plot(list(range(len(values))), values, label=label)
    if args.log_scale:
        plt.yscale("log")
    plt.legend()
    plt.ylabel(args.field)
    plt.xlabel("step")
    plt.show()


def label_for_path(path: str) -> str:
    name, _ = os.path.splitext(os.path.basename(path))
    return name.replace("_", " ")


def read_log(path) -> Dict[str, np.ndarray]:
    result = defaultdict(list)
    with open(path, "r") as f:
        lines = [line for line in f.readlines() if line.startswith("step")]
        for line in lines:
            for field in (x for x in line.split() if "=" in x):
                name, value = field.split("=")
                result[name].append(float(value))
    return {k: np.array(v) for k, v in result.items()}


if __name__ == "__main__":
    main()
