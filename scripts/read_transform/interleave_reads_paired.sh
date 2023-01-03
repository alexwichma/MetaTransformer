#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

### Script to interleave forward and reverse strand fasta files into a single fasta file ###

forward_path=
reverse_path=
output_path=

while getopts “f:r:o:” OPTION
do
    case ${OPTION} in
        f)
            forward_path=${OPTARG}
            ;;
        r)
            reverse_path=${OPTARG}
            ;;
        o)
            output_path=${OPTARG}
            ;;
    esac
done

mkdir -p $(dirname "${output_path}")
echo "Interleaving paired-end reads into a single file..."
seqtk mergepe ${forward_path} ${reverse_path} > ${output_path}
echo "Done"