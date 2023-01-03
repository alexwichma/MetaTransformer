import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('seaborn')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", dest="in_path", type=str, help="Path to abundances csv", required=True)
    parser.add_argument("--out-folder", dest="out_folder", type=str, required=True)
    args = parser.parse_args()
    in_path = args.in_path
    out_folder = args.out_folder

    cmap = plt.get_cmap("Set1")

    df = pd.read_csv(in_path)
    k = df["k"]
    prec_v = df["prec(v)"]
    rec_v = df["rec(v)"]
    loss_v = df["loss(v)"]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    bar_width = 0.22

    x1 = np.arange(0, len(k))
    x2 = [x + bar_width for x in x1]
    x3 = [x + bar_width for x in x2]

    middle = (2 * bar_width) / 2.0
    
    ax1.bar(x1, prec_v, color=cmap(0), width=bar_width, edgecolor="white", label="Precision (Validation)")
    ax1.bar(x2, rec_v, color=cmap(1), width=bar_width, edgecolor="white", label="Recall (Validation)")
    ax2.bar(x3, loss_v, color=cmap(2), width=bar_width, edgecolor="white", label="Loss (Validation)")

    ax1.set_xlabel("kmer size")
    ax1.set_ylabel("Precision / Recall")
    ax1.set_xticks([x + middle for x in range(len(k))])
    ax1.set_xticklabels(k)

    ax2.set_ylabel("Loss")
    ax2.grid(None)

    ax1.legend(bbox_to_anchor=(.12, 1.0, 1., .10), loc=2,ncol=2, borderaxespad=0.)
    ax2.legend(bbox_to_anchor=(0.67, 1.0, 1., .10), loc=2,ncol=1, borderaxespad=0.)
    
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, f"k_mer_prec_rec_loss.pdf"))



if __name__ == "__main__":
    main()