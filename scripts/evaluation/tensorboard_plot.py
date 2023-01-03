import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


plt.style.use('seaborn')


def main():
    parser = argparse.ArgumentParser(description="Plots precision, recal and loss based on tensorboard csv files")
    parser.add_argument("--loss-folder", dest="loss_folder", type=str, required=True)
    parser.add_argument("--precision-folder", dest="precision_folder", type=str, required=True)
    parser.add_argument("--recall-folder", dest="recall_folder", type=str, required=True)
    parser.add_argument("--out-folder", dest="out_folder", type=str, required=True)
    parser.add_argument("--transparent", dest="transparent", action="store_true", default=False)

    args = parser.parse_args()
    loss_folder = args.loss_folder
    precision_folder = args.precision_folder
    recall_folder = args.recall_folder

    transparent = args.transparent
    out_folder = args.out_folder

    cmap = plt.get_cmap("Set1")

    loss_fps = sorted(glob.glob(os.path.join(loss_folder, "*.csv")))
    precision_fps = sorted(glob.glob(os.path.join(precision_folder, "*.csv")))
    recall_fps = sorted(glob.glob(os.path.join(recall_folder, "*.csv")))

    fig, axes = plt.subplots()
    axes.set_xlabel("Step")
    axes.set_ylabel("Validation Loss")

    for index, loss_fp in enumerate(loss_fps):
        filename = os.path.basename(loss_fp)
        filename = filename[1:filename.index(".csv")]
        df = pd.read_csv(loss_fp)
        steps = df["Step"]
        values = df["Value"]
        axes.plot(steps, values, label=filename, color=cmap(index))

    fig2, axes2 = plt.subplots()
    axes2.set_xlabel("Step")
    axes2.set_ylabel("Precision/Recall")

    for index, (precision_fp, recall_fp) in enumerate(zip(precision_fps, recall_fps)):
        filename_prec = os.path.basename(precision_fp)
        filename_prec = filename_prec[1:filename_prec.index(".csv")]
        df_prec = pd.read_csv(precision_fp)
        steps_prec = df_prec["Step"]
        values_prec = df_prec["Value"]

        filename_rec = os.path.basename(recall_fp)
        filename_rec = filename_rec[1:filename_rec.index(".csv")]
        df_rec = pd.read_csv(recall_fp)
        steps_rec = df_rec["Step"]
        values_rec = df_rec["Value"]

        axes2.plot(steps_prec, values_prec, label=filename_prec + " - Precision", color=cmap(index))
        axes2.plot(steps_rec, values_rec, "--", label=filename_rec + " - Recall", color=cmap(index))

    axes.legend()
    axes2.legend()

    fig.savefig(os.path.join(out_folder, f"tb_loss.pdf"), transparent=transparent)
    fig2.savefig(os.path.join(out_folder, f"tb_prec_rec.pdf"), transparent=transparent)

    
if __name__ == "__main__":
    main()