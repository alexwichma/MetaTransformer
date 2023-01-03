import argparse
import glob
import os
import math
from Bio import SeqIO
from multiprocessing import Process, Value, Lock


def convert_worker(file_paths, shared_progress, lock, num_files):
    for file_path in file_paths:
        with open(file_path, 'r') as src_file_handle:
            src_seqs = SeqIO.parse(src_file_handle, "fastq")
            dest_file_path = file_path.replace(".fq", ".fa")
            with open(dest_file_path, "w") as dest_file_handle:
                SeqIO.write(src_seqs, dest_file_handle, "fasta")
        os.remove(file_path)
        lock.acquire()
        try:
            shared_progress.value += 1
            print("Progress [%5d / %5d]" % (shared_progress.value, num_files), end="\r")
        finally:
            lock.release()


def main():
    parser = argparse.ArgumentParser(description="Converts fastq to fasta files.")
    parser.add_argument("-p", dest="path", type=str, help="Path where to look for fastq files", required=True)
    args = parser.parse_args()
    path = args.path
    if os.path.isdir(path):
        file_paths = glob.glob(os.path.join(path, "*.fq"))
    else:
        file_paths = [path]
    num_workers = min(len(file_paths), 64)
    num_files_per_worker = math.floor(len(file_paths) / num_workers)
    progress = Value('i', 0)
    lock = Lock()
    processes = []
    for i in range(num_workers):
        start = i * num_files_per_worker
        end = start + num_files_per_worker if i != num_workers - 1 else len(file_paths)
        args = (file_paths[start:end], progress, lock, len(file_paths))
        p = Process(target=convert_worker, args=args)
        processes.append(p)
        p.start()
    for process in processes:
        p.join()


if __name__ == "__main__":
    main()
