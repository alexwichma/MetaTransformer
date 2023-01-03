import argparse
import glob
import os
import math
from Bio import SeqIO
from multiprocessing import Process, Value, Lock


def label_worker(paths, class_mapping, metadata_dict, level, out_path, progress, lock, max_paths):
    for path in paths:
        filename = os.path.basename(path)
        seqname = filename[(filename.find("_") + 1):filename.find(".")]

        if level == "species":
            taxname, _ = metadata_dict[seqname]
            class_id = class_mapping[taxname]
        else:
            _, taxname = metadata_dict[seqname]
            if taxname == "Unassigned" or taxname not in class_mapping:
                continue
            class_id = class_mapping[taxname]

        out_file_path = os.path.join(out_path, seqname + ".fa")
        out_seqs = []

        with open(path, "r") as file_handle:
            seqs = SeqIO.parse(file_handle, "fasta")
            for seq in seqs:
                header = "lbl|%s|%s|%s" % (class_id, taxname, seq.id)
                seq.id = header
                seq.description = header
                out_seqs.append(seq)

        with open(out_file_path, "w") as out_file_handle:
            SeqIO.write(out_seqs, out_file_handle, "fasta")
    lock.acquire()
    try:
        progress.value += 1
        print("Progress [%5d / %5d]" % (progress.value, max_paths), end="\r")
    finally:
        lock.release()


def main():
    parser = argparse.ArgumentParser(description="Script to label the benchmark gut test data")
    parser.add_argument("-p", dest="path", type=str, help="Input directory containing fasta files", required=True)
    parser.add_argument("-t", dest="class_path", type=str, help="Class mapping path", required=True)
    parser.add_argument("-m", dest="metadata_path", type=str, help="Sequence metadata filepath", required=True)
    parser.add_argument("-o", dest="out_path", type=str, help="Labeled sequence output path", required=True)
    parser.add_argument("-lv", dest="level", type=str, help="Species or genus", required=True)
    args = parser.parse_args()
    path = args.path
    class_mapping_path = args.class_path
    metadata_path = args.metadata_path
    out_path = args.out_path
    level = args.level

    # Fill dict for filename -> taxname mapping
    metadata_dict = {}
    with open(metadata_path, "r") as file_handle:
        lines = file_handle.readlines()[2:]  # skip first two lines
        for line in lines:
            line_split = line.split("\t")
            file_name = line_split[0].rstrip().lstrip()
            species_tax_name = line_split[2].rstrip().lstrip()
            genus_tax_name = line_split[11].rstrip().lstrip()
            metadata_dict[file_name] = [species_tax_name, genus_tax_name]

    # Fill dict for taxname -> taxid mapping
    class_mapping = {}
    with open(class_mapping_path, "r") as file_handle:
        lines = file_handle.readlines()[1:]
        for line in lines:
            class_index, tax_name = line.strip().split("\t")
            class_mapping[tax_name] = class_index
    
    file_paths = glob.glob(os.path.join(path, "*.fa"))
    num_workers = min(len(file_paths), 32)
    num_files_per_worker = math.floor(len(file_paths) / num_workers)
    progress = Value('i', 0)
    lock = Lock()
    processes = []
    for i in range(num_workers):
        start = i * num_files_per_worker
        end = start + num_files_per_worker if i != num_workers - 1 else len(file_paths)
        args = (file_paths[start:end], class_mapping, metadata_dict, level, out_path, progress, lock, len(file_paths))
        p = Process(target=label_worker, args=args)
        processes.append(p)
        p.start()
    for process in processes:
        process.join()


if __name__ == "__main__":
    main()