import argparse
import glob
import os
import numpy as np
import re


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_naturally(arr):
    return sorted(arr, key=alphanum_key)


def aggregate(in_path, num_out_files, out_path):
    label_paths = sort_naturally(glob.glob(os.path.join(in_path, "labels*.npy")))
    read_paths = sort_naturally(glob.glob(os.path.join(in_path, "reads*.npy")))
    assert len(label_paths) == len(read_paths)
    assert num_out_files > 0
    assert len(label_paths) > num_out_files
    first_reads = np.load(read_paths[0])

    # load last one to check for diverging read count, not needed afterward
    last_reads = np.load(read_paths[-1])
    last_num_reads = last_reads.shape[0]

    print(first_reads.shape)
    # shape constants
    num_files = len(label_paths)
    reads_per_file = first_reads.shape[0]
    read_length = first_reads.shape[1]

    # create numpy arrays, add last file as exception
    num_complete_reads = (num_files - 1) * reads_per_file + last_num_reads
    reads_per_out_file = num_complete_reads // num_out_files
    reads_last_out_file = reads_per_out_file + (num_complete_reads % reads_per_out_file)

    # counters for file and array positions
    curr_in_file_idx = 0
    curr_in_file_pos = 0
    curr_read_file = None
    curr_label_file = None

    for i in range(num_out_files):
        num_reads = reads_per_out_file if i != num_out_files - 1 else reads_last_out_file
        out_reads = np.zeros((num_reads, read_length), dtype=np.int32)
        out_labels = np.zeros((num_reads,), dtype=np.int16)
        curr_out_file_pos = 0

        while curr_out_file_pos < (num_reads - 1):
            if curr_read_file is None and curr_label_file is None:
                curr_read_file = np.load(read_paths[curr_in_file_idx])
                curr_label_file = np.load(label_paths[curr_in_file_idx])
                curr_in_file_pos = 0
            # In-file does fit completely into out-file, in-file exhausted
            if (curr_read_file.shape[0] - curr_in_file_pos) <= (num_reads - curr_out_file_pos):
                elems = curr_read_file.shape[0] - curr_in_file_pos
                out_end = curr_out_file_pos + elems
                in_end = curr_in_file_pos + elems
                out_reads[curr_out_file_pos:out_end] = curr_read_file[curr_in_file_pos:in_end]
                out_labels[curr_out_file_pos:out_end] = curr_label_file[curr_in_file_pos:in_end]
                curr_out_file_pos = out_end
                curr_read_file = None
                curr_label_file = None
                curr_in_file_idx += 1
            # In-file does not fit completely into out file, out-file exhausted
            else:
                elems = num_reads - curr_out_file_pos
                out_end = num_reads
                in_end = curr_in_file_pos + elems
                out_reads[curr_out_file_pos:out_end] = curr_read_file[curr_in_file_pos:in_end]
                out_labels[curr_out_file_pos:out_end] = curr_label_file[curr_in_file_pos:in_end]
                curr_in_file_pos = in_end
                curr_out_file_pos = out_end
                
        np.save(os.path.join(out_path, "reads_%d.npy" % (i)), out_reads)
        np.save(os.path.join(out_path, "labels_%d.npy" % (i)), out_labels)


def main():
    parser = argparse.ArgumentParser(description="Aggregates multiple .npy files into a single one")
    parser.add_argument("-p", dest="in_path", type=str, help="Path where to look for npy files", required=True)
    parser.add_argument("-o", dest="out_path", type=str, help="Path where to save aggregated files", required=True)
    parser.add_argument("-n", dest="num_files", type=int, help="Number of files to aggregate to", default=1)
    args = parser.parse_args()
    aggregate(args.in_path, args.num_files, args.out_path)


if __name__ == "__main__":
    main()