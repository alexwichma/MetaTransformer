"""
Script that is used to calculate the class weights for the loss function in multi-level classification
"""


from __future__ import absolute_import, division, print_function

import numpy as np
import os
from glob import glob
import argparse

from utils.taxonomy_utils import str_to_rank


def count_class_occurences(read_paths, ranks, num_classes_per_rank, class_label_idxs):
    counts = {}
    for rank, class_per_rank in zip(ranks, num_classes_per_rank):
        counts[rank] = np.zeros(class_per_rank, dtype=np.float32)
    for read_path in read_paths:
        with open(read_path, "r") as read_file:
            for line in read_file:
                line = line.strip()
                if line.startswith(">"):
                    line_split = line.split("|")
                    # only actually observe first label and skip rest of file
                    # since we expect that reads are generated evenly on the lowest
                    # ranks anyways.
                    for rank, label_idx in zip(ranks, class_label_idxs):
                        class_id = int(line_split[label_idx])
                        counts[rank][class_id] += 1
                    break
    return counts


def weight_counts(counts):
    weightings = {}
    for rank, counts_per_rank in counts.items():
        inversed = np.divide(1.0, counts_per_rank, where=counts_per_rank != 0)
        normalized = inversed / inversed.sum()
        weightings[rank] = normalized
    return weightings


def calculate_class_weightings(data_path, ranks, num_classes_per_rank, label_idxs):
    if os.path.isdir(data_path):
        read_paths = glob(os.path.join(data_path, "*.fa"))
    else:
        read_paths = [data_path]
    # at maximum as many processes as read paths make sense
    class_counts = count_class_occurences(read_paths, ranks, num_classes_per_rank, label_idxs)
    return weight_counts(class_counts)

def main():
    parser = argparse.ArgumentParser(description="Script to generate class weightings for genome train data.")
    parser.add_argument("--seq-path", dest="sequence_path", type=str, help="Path to genome sequences", required=True)
    parser.add_argument("--save-path", dest="save_path", type=str, help="Path where to save class weightings", required=True)
    parser.add_argument("--ranks", dest="ranks", type=str, help="Comma-separated list of ranks (need to be the string version of TaxonomicRank enum)", required=True)
    parser.add_argument("--classes", dest="classes", type=str, help="Comma-separated list of number of classes per rank", required=True)
    parser.add_argument("--idxs", dest="idxs", type=str, help="Comma-separated list where to find the corresponding labels in the genome header", required=True)
    args = parser.parse_args()
    sequence_path = args.sequence_path
    save_path = args.save_path
    ranks = [str_to_rank(s) for s in args.ranks.split(",")]
    num_classes = [int(n) for n in args.classes.split(",")]
    label_idxs = [int(i) for i in args.idxs.split(",")]
    weightings = calculate_class_weightings(sequence_path, ranks, num_classes, label_idxs)
    np.save(save_path, weightings)

if __name__ == "__main__":
    main()