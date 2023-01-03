#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

### Script to inverleave paired-end reads and their reverse complement into a single fasta file ###
### This means that per paired-end read, four reads are interleaved into the output fasta file  ###

forward_path=
reverse_path=
output_path=

while (( "$#" )); do
    case "$1" in
        --forward-path)
            forward_path=$2
            shift 2
            ;;
        --reverse-path)
            reverse_path=$2
            shift 2
            ;;
        --output-path)
            output_path=$2
            shift 2
            ;;
        *)
            echo "Unsupported argument $1"
            shift
            ;;
    esac
done

forward_file_name=${forward_path##*/}
reverse_file_name=${reverse_path##*/}

output_dir=$(dirname "${output_path}")

mkdir -p $output_dir
echo "Creating reverse complements of forward and reverse strand"
seqtk seq -r ${forward_path} > $output_dir/$forward_file_name.rc
seqtk seq -r ${reverse_path} > $output_dir/$reverse_file_name.rc
echo "Interleaving sequences"
seqtk mergepe $forward_path $reverse_path > $output_dir/interleaved_1.fa
seqtk mergepe $output_dir/$forward_file_name.rc $output_dir/$reverse_file_name.rc > $output_dir/interleaved_2.fa
seqtk mergepe $output_dir/interleaved_1.fa $output_dir/interleaved_2.fa > $output_path
rm $output_dir/$forward_file_name.rc
rm $output_dir/$reverse_file_name.rc
rm $output_dir/interleaved_1.fa
rm $output_dir/interleaved_2.fa
echo "Done"
