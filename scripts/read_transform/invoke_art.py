import argparse
import glob
import os
from Bio import SeqIO
import subprocess
from copy import deepcopy
import operator


def determine_genome_length(seqs):
    seqs_len = 0
    for seq in seqs:
        seqs_len += len(seq.seq)
    return seqs_len


def find_genome_lengths(in_dir):
    paths = glob.glob(os.path.join(in_dir, "*.fa"))
    result = {}
    for path in paths:
        filename = os.path.basename(path)
        with open(path, "r") as handle:
            seqs = SeqIO.parse(handle, "fasta")
            seqs_len = determine_genome_length(seqs)
            result[filename] = seqs_len
    return result


def determine_coverages(genome_lengths, cov_factor):
    max_len = max(genome_lengths.items(), key=operator.itemgetter(1))[1]
    coverages = {}
    for filename, genome_len in genome_lengths.items():
        coverages[filename] = str(int(round((max_len / genome_len) * cov_factor)))
    return coverages


def invoke_processes(art_path, frag_mean, frag_dev, seq_sys, read_len, job_contents, parallel=64):
    BASE_ART_CMD = [
        art_path, "--paired", "--noALN", "--quiet", "--mflen",
        frag_mean, "--sdev", frag_dev, "--seqSys", seq_sys, "--len", read_len
    ]
    for i in range(0, len(job_contents), parallel):
        procs = []
        for job_content in job_contents[i:(i + parallel)]:
            ART_CMD = deepcopy(BASE_ART_CMD) + job_content
            procs.append(subprocess.Popen(ART_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        for proc in procs:
            proc.wait()


def generate_input(in_dir, out_dir, coverages=None, rcount="-1"):
    assert coverages is not None or rcount != "-1"
    paths = glob.glob(os.path.join(in_dir, "*.fa"))
    result = []
    for i, path in enumerate(paths):
        out_path = os.path.join(out_dir, "reads_%d" % i)
        if rcount == "-1":
            cov = coverages[os.path.basename(path)]   
            result.append(["--in", path, "--fcov", cov, "--out", out_path])
        else:
            result.append(["--in", path, "--rcount", rcount, "--out", out_path])
    return result


def main():
    parser = argparse.ArgumentParser(description="Calculates per-genome coverages to generate balanced number of reads with ART")
    parser.add_argument("-i", dest="path", type=str, help="Path to reference sequences", required=True)
    parser.add_argument("-l", dest="read_len", type=str, help="Read length", required=True)
    parser.add_argument("-r", dest="read_count", type=str, help="Number of reads to generate", required=True)
    parser.add_argument("-s", dest="seq_sys", type=str, help="Sequencing system", required=True)
    parser.add_argument("-m", dest="frag_mean", type=str, help="Mean fragment length", required=True)
    parser.add_argument("-d", dest="frag_deviation", type=str, help="Fragment deviation", required=True)
    parser.add_argument("-c", dest="cov_factor", type=int, help="Coverage factor", required=True)
    parser.add_argument("-t", dest="art_tool_path", type=str, help="Art tool path", required=True)
    parser.add_argument("-o", dest="out_path", type=str, help="Out path for generated reads", required=True)
    args = parser.parse_args()
    path = args.path
    read_len = args.read_len
    read_count = args.read_count
    seq_sys = args.seq_sys
    frag_mean = args.frag_mean
    frag_deviation = args.frag_deviation
    cov_factor = args.cov_factor
    art_tool_path = args.art_tool_path
    out_path = args.out_path

    inputs = []
    if read_count == "-1":
        print("Using coverage factor for read generation. Generating as many reads as needed to achieve coverage of %s for the biggest genome. For smaller genomes, additional reads by factor (longest_genome_len / genome_len) will be generated." % cov_factor)
        genome_lengths = find_genome_lengths(path)
        print("Genome lengths:")
        print(sorted(genome_lengths.items(), key=lambda item: item[1]))
        coverages = determine_coverages(genome_lengths, cov_factor)
        print("Determined coverages:")
        print(sorted(coverages.items(), key=lambda item: item[1]))
        inputs = generate_input(path, out_path, coverages=coverages)
    else:
        print("Using read count for read generation. %s reads per genome (=file) will be generated." % read_count)
        inputs = generate_input(path, out_path, rcount=read_count)
    invoke_processes(art_tool_path, frag_mean, frag_deviation, seq_sys, read_len, inputs)


if __name__ == "__main__":
    main()
