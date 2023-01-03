import argparse
from glob import glob
import os
import pandas as pd
from functools import reduce

def merge_dfs(file_paths, grouping_col, pattern_len):
    dfs = [pd.read_csv(file_path, index_col=0) for file_path in file_paths]

    names = ["Taxon"]
    for i in file_paths:
        tmp = i.split("/")
        tmp = tmp[len(tmp)-1][:-pattern_len]
        names.append(tmp)

    df_merged_prediction = reduce(lambda  left,right: pd.merge(left, right.iloc[:,[0,1]], on=['Taxon'], how='outer'), dfs)
    df_merged_prediction = df_merged_prediction.drop(columns=["Norm_Prediction"] )
    df_merged_prediction.columns = names

    df_merged_normprediction = reduce(lambda  left,right: pd.merge(left, right.iloc[:,[0,2]], on=['Taxon'], how='outer'), dfs)

    df_merged_normprediction = df_merged_normprediction.drop(columns=["Prediction"] )
    df_merged_normprediction.columns = names

    return df_merged_prediction, df_merged_normprediction


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

    pattern = os.path.join(in_path, "*" + str(file_name_pattern))
    file_paths = glob(pattern)
    
    assert len(file_paths)>0, "No matching files or wrong pattern."
    
    pattern_len = len (file_name_pattern)
    result, result_norm = merge_dfs(file_paths, grouping_col, pattern_len)

    result.to_csv(os.path.join(out_path, "abundance.csv"))
    result_norm.to_csv(os.path.join(out_path, "norm_abundance.csv"))


if __name__ == "__main__":
    main()

