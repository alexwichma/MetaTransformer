import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

prec_rec_dm_path = "/share/ebuschon/data/dm_original_result_data/mock/precision-recall.csv"
prec_rec_our_path = "/share/ebuschon/evaluation_mock/elstmatt_v12/avg_read_lv_results.csv"


prec_rec_dm_df = pd.read_csv(prec_rec_dm_path)
prec_rec_dm_df = prec_rec_dm_df[["Confidence score", "Specificity (average)", "Sensitivity (average)"]]
prec_rec_dm_df = prec_rec_dm_df.rename(columns={"Specificity (average)": "Precision (Original)", "Sensitivity (average)": "Recall (Original)", "Confidence score": "threshold"})

prec_rec_our_df = pd.read_csv(prec_rec_our_path)
prec_rec_our_df = prec_rec_our_df[["threshold", "precision", "recall"]]
prec_rec_our_df = prec_rec_our_df.rename(columns={"precision": "Precision (Ours)", "recall": "Recall (Ours)"})

joined = pd.merge(prec_rec_our_df, prec_rec_dm_df, on="threshold")

xticks = np.array([round(0.05 * x, ndigits=2) for x in range(0, 21)])

fig, axes = plt.subplots()
axes.set_xlabel("Threshold")
axes.set_ylabel("Read level precision/recall")
axes.plot(xticks, joined["Precision (Ours)"].to_list(), color="red", label="Precision (Ours)")
axes.plot(xticks, joined["Recall (Ours)"].to_list(), color="blue", label="Recall (Ours)")
axes.plot(xticks, joined["Precision (Original)"].to_list(), color="green", label="Precision (Original)")
axes.plot(xticks, joined["Recall (Original)"].to_list(), color="purple", label="Recall (Original)")
axes.legend()

fig.savefig(os.path.join("../../result_images", f"genus_prec_recall_comp.pdf"), transparent=False)