import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Splits a single .npy into multiple npy files")
    parser.add_argument("-p", dest="in_path", type=str, help="Path where to look for .npy files", required=True)
    parser.add_argument("-o", dest="out_path", type=str, help="Path where to save .npy output files", required=True)
    parser.add_argument("-s", dest="num_split", type=int, help="Number of files to split to", default=35)
    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    num_split = args.num_split

    reads_path = os.path.join(in_path, "reads.npy")
    labels_path = os.path.join(in_path, "labels.npy")
    reads = np.load(reads_path)
    labels = np.load(labels_path)

    num_elements = reads.shape[0] // num_split
    num_remaining_elements = reads.shape[0] % num_split

    for i in range(num_split):
        start = i * num_elements
        end = start + num_elements
        # Last file needs to handle more elements than the rest
        if i == num_split - 1:
            end += num_remaining_elements
        chunk_reads = reads[start:end]
        chunk_labels = labels[start:end]
        out_path_read = os.path.join(out_path, "reads_%d.npy" % (i))
        out_path_label = os.path.join(out_path, "labels_%d.npy" % (i))
        np.save(out_path_read, chunk_reads)
        np.save(out_path_label, chunk_labels)

    os.remove(reads_path)
    os.remove(labels_path)


if __name__ == "__main__":
    main()
