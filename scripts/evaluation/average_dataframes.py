import argparse
from glob import glob
import os
import pandas as pd


def merge_dfs(file_paths, grouping_col):
    dfs = [pd.read_csv(file_path, index_col=0) for file_path in file_paths]
    concat = pd.concat(dfs, ignore_index=True)
    return concat.groupby([grouping_col]).mean().reset_index()


def main():
    parser = argparse.ArgumentParser(description="Merges different dataframes from subdirectories based on a file name and a grouping column")
    parser.add_argument("--in-path", dest="in_path", type=str, help="Path to mock communities directory", required=True)
    parser.add_argument("--file-name-pattern", dest="file_name_pattern", type=str, help="Name of the files in each subdirectory", required=True)
    parser.add_argument("--grouping-col", dest="grouping_col", type=str, help="Column on which to group", required=True)
    parser.add_argument("--out-path", dest="out_path", type=str, help="Path where to save the merged dataframe", required=True)
    args = parser.parse_args()

    in_path = args.in_path
    file_name_pattern = args.file_name_pattern
    grouping_col = args.grouping_col
    out_path = args.out_path

    file_paths = glob(os.path.join(in_path, "*", file_name_pattern))
    result = merge_dfs(file_paths, grouping_col)
    result = result.round(4)
    result.to_csv(out_path)


if __name__ == "__main__":
    main()