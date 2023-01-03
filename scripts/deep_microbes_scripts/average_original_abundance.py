import os
import glob
import pandas as pd


original_abundance_folder = "/share/ebuschon/data/dm_original_mock_abundances"
our_abundance_folder = "/share/ebuschon/evaluation_mock/elstmatt_v12"
out_folder = "/share/ebuschon/data/dm_original_mock_abundances/merged"

original_abundance_fps = sorted(glob.glob(os.path.join(original_abundance_folder, "*.txt")))
our_abundance_fps = sorted(glob.glob(os.path.join(our_abundance_folder, "mock*/abundances.csv")))

for idnex, (original_fp, our_fp) in enumerate(zip(original_abundance_fps, our_abundance_fps)):
    our_df = pd.read_csv(our_fp)
    our_df = our_df.rename(columns={"Prediction": "Prediction(Ours)"})
    original_df = pd.read_csv(original_fp, sep="\t")
    original_df = original_df[["DeepMicrobes", "Genus"]]
    original_df = original_df.rename(columns={"DeepMicrobes": "Prediction(Original)", "Genus": "Taxon"})
    joined = pd.merge(our_df, original_df, "left", on="Taxon")

    joined.to_csv(os.path.join(out_folder, f"abundance_{idnex}.csv"))