import argparse
import glob
import os
import math
from Bio import SeqIO
from multiprocessing import Process, Value, Lock


def label_worker(paths, tax_dict, metadata_dict, out_path, progress, lock, max_paths):
    for path in paths:
        filename = os.path.basename(path)
        out_seqs = []
        with open(path, "r") as file_handle:
            seqs = SeqIO.parse(file_handle, "fasta")
            for seq in seqs:
                seq_id_pref = seq.id[:(seq.id.find("."))]
                genus_name = metadata_dict[seq_id_pref]
                genus_id = tax_dict[genus_name]
                header = "label|%s|%s|%s" % (genus_id, genus_name, seq.id)
                seq.id = header
                seq.description = header
                out_seqs.append(seq)
            out_file_path = os.path.join(out_path, filename + ".fa")
            with open(out_file_path, "w") as out_file_handle:
                SeqIO.write(out_seqs, out_file_handle, "fasta")
        lock.acquire()
        try:
            progress.value += 1
            print("Progress [%5d / %5d]" % (progress.value, max_paths), end="\r")
        finally:
            lock.release()


def main():
    parser = argparse.ArgumentParser(description="Script to label the benchmark mock test data")
    parser.add_argument("-p", dest="path", type=str, help="Input directory containing fasta files", required=True)
    parser.add_argument("-t", dest="mapping_path", type=str, help="Path to taxonomy file", required=True)
    parser.add_argument("-m", dest="metadata_path", type=str, help="Path to mock community metadata file", required=True)
    parser.add_argument("-o", dest="out_path", type=str, help="Labeled sequence output path", required=True)
    args = parser.parse_args()
    path = args.path
    mapping_path = args.mapping_path
    metadata_path = args.metadata_path
    out_path = args.out_path

    # Fill dict for sequence prefix -> taxname mapping
    metadata_dict = {}
    with open(metadata_path, "r") as file_handle:
        lines = file_handle.readlines()[1:]  # skip first line
        for line in lines:
            line_split = line.split(",")
            sequence_pref = line_split[0].rstrip().lstrip()
            tax_name = line_split[2].rstrip().lstrip()
            metadata_dict[sequence_pref] = tax_name

    # Fill dict for taxname -> taxid mapping
    tax_dict = {}
    with open(mapping_path, "r") as file_handle:
        lines = file_handle.readlines()[1:]
        for index, genus_line in enumerate(lines):
            tax_name = genus_line.split("\t")[1].rstrip().lstrip()
            tax_dict[tax_name] = index

    file_paths = glob.glob(os.path.join(path, "*.fa"))
    num_workers = min(len(file_paths), 5)
    num_files_per_worker = math.floor(len(file_paths) / num_workers)
    progress = Value('i', 0)
    lock = Lock()
    processes = []
    for i in range(num_workers):
        start = i * num_files_per_worker
        end = start + num_files_per_worker if i != num_workers - 1 else len(file_paths)
        args = (file_paths[start:end], tax_dict, metadata_dict, out_path, progress, lock, len(file_paths))
        p = Process(target=label_worker, args=args)
        processes.append(p)
        p.start()
    for process in processes:
        p.join()


if __name__ == "__main__":
    main()