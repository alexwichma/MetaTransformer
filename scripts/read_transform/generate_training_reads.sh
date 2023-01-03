#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

#### Script to generate training reads from a collection of raw training genome FASTA files ###

readonly ART_TOOL_PATH="../../bin/art_illumina"
readonly INVOKE_ART_PATH="./invoke_art.py"
readonly SEQ_SHUF_TOOL_PATH="../../bin/seq-shuf"
readonly FQ_2_FA_PATH="./fq2fa.py"
readonly FA_2_KMER_PATH="./fa2kmer.py"
readonly AGGREGATE_FILES_PATH="../numpy_transform/aggregate_npy_files.py"

# Sequencing system parameters change them as needed
readonly READ_LENGTH=150
readonly SEQ_SYS="HS25"
readonly MEAN_FRAGMENT_LENGTH=400
readonly FRAGMENT_DEVIATION=50
# How many lines each sequence spans this is needed to properly split the files into multiple files
readonly LINES_PER_SEQ=4

fasta_in_path=
fasta_out_path=
num_file_split=64

# use GENOME_COVERAGE, except if user provides num_reads
coverage=3

while getopts "i:c:o:n:" OPTION
do
    case ${OPTION} in
        i)
            fasta_in_path=${OPTARG}
            ;;
        c)
            coverage=${OPTARG}
            ;;
        o)
            fasta_out_path=${OPTARG}
            ;;
        n)  
            num_file_split=${OPTARG}
            ;;
    esac
done

# Create tmp dir
mkdir -p ${fasta_out_path}/tmp

# Generate reads. Use a python script to determine per-genome coverages.
# In this manner, a balanced number of reads per class can be generated.
echo "Generating reads with ART tool."
python3 ${INVOKE_ART_PATH} \
    -i ${fasta_in_path} \
    -l ${READ_LENGTH} \
    -s ${SEQ_SYS} \
    -r -1 \
    -m ${MEAN_FRAGMENT_LENGTH} \
    -d ${FRAGMENT_DEVIATION} \
    -c ${coverage} \
    -t ${ART_TOOL_PATH} \
    -o ${fasta_out_path}/tmp

split_data_to_chunks () {
    in_path=$1
    in_file=$2
    num_files=$3
    chunk_out_name=$4
    chunk_suffix=$5
    num_lines_fq=$(cat ${in_path}/${in_file} | wc -l)
    split_size_fq=$((${num_lines_fq} / ${LINES_PER_SEQ} / ${num_file_split} * ${LINES_PER_SEQ}))
    split -d -l ${split_size_fq} --additional-suffix=${chunk_suffix} ${in_path}/${in_file} ${in_path}/${chunk_out_name}
    rm ${tmp_path}/${in_file}
}

# Temp dir to work in
tmp_path=${fasta_out_path}/tmp

# Two-stranded reads from generator. Merge them first. Then split them into multiple files for python multiprocessing
echo "Merging files."
cat ${tmp_path}/reads*.fq > ${tmp_path}/merged_reads.fq
echo "Deleting original chunks."
rm ${tmp_path}/reads*.fq
echo "Splitting data into ${num_file_split} chunks."
split_data_to_chunks ${tmp_path} merged_reads.fq ${num_file_split} read_chunk_ .fq
echo "Converting FASTQ to FASTA format."
python3 ${FQ_2_FA_PATH} -p ${tmp_path}
echo "Shuffling reads"
cat ${tmp_path}/*.fa | seq-shuf > ${fasta_out_path}/reads.fa
echo "Deleting temporary files"
rm  -r ${tmp_path}
echo "Done"

exit 0