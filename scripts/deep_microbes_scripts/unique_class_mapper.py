import argparse
import pandas as pd


def main(args):
    in_file = args.in_file
    out_file = args.out_file
    df = pd.read_csv(in_file, sep="\t")
    with open(out_file, "w") as file_handle:
        for col in df:
            file_handle.write("[{}]\n".format(col))
            unique_labels = sorted(list(df[col].dropna().unique()))
            for index, label in enumerate(unique_labels):
                file_handle.write("{}\t{}\n".format(index, label))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", dest="in_file", type=str, required=True)
    parser.add_argument("--out-file", dest="out_file", type=str, required=True)
    args = parser.parse_args()
    main(args)