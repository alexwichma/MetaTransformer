import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


plt.style.use('seaborn')


def main():
    parser = argparse.ArgumentParser(description="Plots the memory requirements of different architectures")
    parser.add_argument("--in-path", dest="in_path", type=str, help="Path to abundances csv", required=True)
    parser.add_argument("--out-folder", dest="out_folder", type=str, required=True)
    parser.add_argument("--transparent", dest="transparent", action="store_true", default=False)
    args = parser.parse_args()
    in_path = args.in_path
    transparent = args.transparent
    out_folder = args.out_folder

    df = pd.read_csv(in_path)
    df = df.fillna("")

    model_names = df["Modelname"]
    modifications = df["Modification"]
    training_memories = df["TrainingMemory(GB)"]
    inference_memories = df["InferenceMemory(GB)"]
    training_memories_sparse_amp = df["TrainingMemory(GB)SparseAmp"]
    inference_memores_sparse_amp = df["InferenceMemory(GB)SparseAmp"]

    cmap = plt.get_cmap("Set1")

    labels = []

    for model_name, modification in zip(model_names, modifications):
        if len(modification) > 0:
            label = model_name + "\n" + r"${}$".format(modification)
            labels.append(label)
        else:
            labels.append(model_name)

    fig, axes = plt.subplots()
    
    bar_width = 0.15

    x1 = np.arange(0, len(training_memories))
    x2 = [x + bar_width for x in x1]
    x3 = [x + bar_width for x in x2]
    x4 = [x + bar_width for x in x3]
    
    axes.bar(x1, training_memories, color=cmap(0), width=bar_width, edgecolor="white", label="Training")
    axes.bar(x2, inference_memories, color=cmap(1), width=bar_width, edgecolor="white", label="Inference")
    axes.bar(x3, training_memories_sparse_amp, color=cmap(2), width=bar_width, edgecolor="white", label="Training(+Sparse+Amp)")
    axes.bar(x4, inference_memores_sparse_amp, color=cmap(3), width=bar_width, edgecolor="white", label="Inference(+Sparse+Amp)")
    

    axes.set_xlabel("Model")
    axes.set_ylabel("Memory consumption (in gigabyte)")
    axes.set_xticks([x + 1.5 * bar_width for x in range(len(labels))])
    axes.set_xticklabels(labels)

    axes.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=4, mode="expand", borderaxespad=0.)
    
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "memory_requirements.pdf"), transparent=transparent)



if __name__ == "__main__":
    main()