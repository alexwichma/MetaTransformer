import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


plt.style.use('seaborn')


def main():
    parser = argparse.ArgumentParser(description="Creates a plot comparing abundance predictions")
    parser.add_argument("--in-path", dest="in_path", type=str, help="Path to file containing abundances", required=True)
    parser.add_argument("--arch-name", dest="arch_name", type=str, help="Name of the architecture", required=True)
    parser.add_argument("--out-folder", dest="out_folder", type=str, help="Path where to save image", required=True)
    parser.add_argument("--transparent", dest="transparent", action="store_true", help="Whether the image has transparent background or not", default=False)
    args = parser.parse_args()
    in_path = args.in_path
    arch_name = args.arch_name
    transparent = args.transparent
    out_folder = args.out_folder

    pct_t = 2.5
    df = pd.read_csv(in_path)
    df_filtered_ge = df[df["GroundTruth"] >= pct_t]
    df_filtered_le = df[df["GroundTruth"] < pct_t]
    le_gt_agg = df_filtered_le["GroundTruth"].sum()
    le_pred_ours_agg = df_filtered_le["Prediction(Ours)"].sum()
    le_pred_original_agg = df_filtered_le["Prediction(Original)"].sum()

    fig, axes = plt.subplots()
    
    bar_width = 0.275
    
    predictions_ours = [*df_filtered_ge["Prediction(Ours)"].tolist(), le_pred_ours_agg]
    predictions_original = [*df_filtered_ge["Prediction(Original)"].tolist(), le_pred_original_agg]
    gts = [*df_filtered_ge["GroundTruth"].tolist(), le_gt_agg]
    y_labels = [*df_filtered_ge["Taxon"].tolist(), f'Rest (<{pct_t}%)']

    y1 = np.arange(0, len(y_labels))
    y2 = [y + bar_width for y in y1]
    y3 = [y + bar_width for y in y2]
    
    axes.barh(y1, predictions_ours, height=bar_width, color="red", edgecolor="white", label="Abundance (ours)")
    axes.barh(y2, predictions_original, color="green", height=bar_width, edgecolor="white", label="Abundance (original)")
    axes.barh(y3, gts, color="blue", height=bar_width, edgecolor="white", label="Ground-truth")

    axes.set_ylabel("Taxon")
    axes.set_xlabel("Normalized abundance (in %)")
    axes.set_yticks([(y + bar_width)for y in y1])
    axes.set_yticklabels(y_labels)

    axes.legend(bbox_to_anchor=(0.1, 1.02, 1., .102), loc=3,ncol=3, mode=None, borderaxespad=0.)
    
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, f"abundance_comp_{arch_name}.pdf"), transparent=transparent)


if __name__ == "__main__":
    main()