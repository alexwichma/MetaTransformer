import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


plt.style.use('seaborn')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", dest="in_file", type=str, required=True)
    parser.add_argument("--out-folder", dest="out_folder", type=str, required=True)
    parser.add_argument("--transparent", dest="transparent", action="store_true", default=False)
    args = parser.parse_args()
    transparent = args.transparent
    in_file = args.in_file
    out_folder = args.out_folder

    df = pd.read_csv(in_file)
    df = df.fillna("")
    model_names = df["Modelname"]
    modifications = df["Modification"]
    time = df["Trainingtime(hours)"]
    labels = []
    for model_name, modification in zip(model_names, modifications):
        if len(modification) > 0:
            labels.append(f"{model_name}\n({modification})")
        else:
            labels.append(model_name)

    xticks = np.arange(0, len(labels))

    fig, axes = plt.subplots()
    axes.bar(labels, time)

    axes.set_xticks(xticks)
    axes.set_xticklabels(labels)
    axes.set_xlabel("Model", labelpad=10)
    axes.set_ylabel("Training time (in hours)")

    axes.set_title("Comparison of training times")

    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "training_time.pdf"), transparent=transparent)


if __name__ == "__main__":
    main()