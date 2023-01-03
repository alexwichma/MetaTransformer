import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.style.use('seaborn')


def main():
    parser = argparse.ArgumentParser(description="Plots a precision/recall plot for threshold selection")
    parser.add_argument("--in-path", dest="in_path", type=str, help="Path to file containing read-level results", required=True)
    parser.add_argument("--arch-name", dest="arch_name", type=str, required=True)
    parser.add_argument("--out-folder", dest="out_folder", type=str, required=True)
    parser.add_argument("--transparent", dest="transparent", action="store_true", default=False)

    args = parser.parse_args()
    in_path = args.in_path
    arch_name = args.arch_name
    transparent = args.transparent
    out_folder = args.out_folder

    t = 0.95
    df = pd.read_csv(in_path)
    precision = df["precision"].to_numpy()
    recall = df["recall"].to_numpy()
    first_over_t_idxs = np.where(precision >= t)
    first_over_t_idx = -1
    if len(first_over_t_idxs) == 0:
        first_over_t_idx = np.argmax(precision)
    else:
        first_over_t_idx = first_over_t_idxs[0][0]
    used_threshold_prec = precision[first_over_t_idx]
    used_threshold_prec_arr = np.repeat(used_threshold_prec, len(precision))
    xticks = np.array([round(0.05 * x, ndigits=2) for x in range(0, 21)])

    fig, axes = plt.subplots()
    axes.set_xlabel("Threshold")
    axes.set_ylabel("Read level precision/recall")
    axes.plot(xticks, precision, color="red", label="Precision")
    axes.plot(xticks, recall, color="blue", label="Recall")
    axes.plot(xticks, used_threshold_prec_arr, "-", color="grey", label=f"Precision at least {t}")
    axes.legend()
    axes.set_title(arch_name)

    fig.savefig(os.path.join(out_folder, f"threshold_selection_{arch_name}.png"), transparent=transparent)


if __name__ == "__main__":
    main()