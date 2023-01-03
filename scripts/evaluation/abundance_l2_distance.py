import numpy as np
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Calculates the L2-distance for an abundance prediction")
    parser.add_argument("--in-path", dest="in_path", type=str, help="Path to abundance prediction csv file", required=True)
    parser.add_argument("--predicted-col-name", dest="p_col", type=str, help="Name of the column containing the prediction", default="Prediction")
    parser.add_argument("-gt-col-name", dest="gt_col", type=str, help="Name of the column containing the ground-truth", default="GroundTruth")
    parser.add_argument("--out-path", dest="out_path", type=str, help="Path where to write the text file containing the L2-distance", required=True)
    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    p_col = args.p_col
    gt_col = args.gt_col
    
    df = pd.read_csv(in_path)
    num_predictions = df.shape[0]
    errors = (df[p_col] - df[gt_col]) / 100.0
    errors = (errors ** 2).sum()
    errors = errors / num_predictions
    errors = np.sqrt(errors)

    with open(out_path, "w") as f_handle:
        f_handle.write(str(errors))
        f_handle.write("\n")


if __name__ == "__main__":
    main()