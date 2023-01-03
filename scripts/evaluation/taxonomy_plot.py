import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob


plt.style.use('seaborn')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", dest="in_path", type=str, help="Path to taxonomy directory", required=True)
    parser.add_argument("--out-folder", dest="out_folder", type=str, help="Path to out folder", required=True)
    parser.add_argument("--transparent", dest="transparent", action="store_true", default=False)

    args = parser.parse_args()
    in_path = args.in_path
    out_folder = args.out_folder
    transparent = args.transparent

    dfs = []

    paths = glob.glob(os.path.join(in_path, "taxonomy_*"))

    for path in paths:
        df = pd.read_csv(path, sep="\t")
        df = df["Genus"]
        dfs.append(df)
    
    df_complete = pd.concat(dfs)
    unique_counts = df_complete.value_counts().to_dict()
    labels = [k for k,_ in unique_counts.items()]
    counts = [v for _, v in unique_counts.items()]
    n = 10
    top_n_labels = labels[:n]
    top_n_counts = counts[:n]
    rest_label = "Rest"
    rest_count = sum(counts[n:])

    plot_labels = [*top_n_labels, rest_label]
    plot_counts = [*top_n_counts, rest_count]

    fig, axes = plt.subplots()
    axes.barh(plot_labels, plot_counts)

    axes.set_ylabel("Genus")
    axes.set_xlabel("Count")

    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "taxonomy_genus_count.pdf"), transparent=transparent)



if __name__ == "__main__":
    main()