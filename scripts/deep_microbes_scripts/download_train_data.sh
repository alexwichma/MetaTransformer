#!/usr/bin/env bash

# Path to train data ftp server: ftp://ftp.ebi.ac.uk/pub/databases/metagenomics/umgs_analyses
DATA_PATH="../../data/deep_microbes_data/train_sequences/"
LABEL_PATH="../../data/deep_microbes_data/train_labels/"

if [ ! -d $DATA_PATH ]; then
    mkdir -p $DATA_PATH
fi

if [ ! -d $LABEL_PATH ]; then
    mkdir -p $LABEL_PATH
fi

wget -r -N --no-parent -nH -nd "ftp://ftp.ebi.ac.uk/pub/databases/metagenomics/umgs_analyses/umgs/genomes/" -P $DATA_PATH
wget -r -N --no-parent -nH -nd "ftp://ftp.ebi.ac.uk/pub/databases/metagenomics/umgs_analyses/hgr/genomes/" -P $DATA_PATH
wget "ftp://ftp.ebi.ac.uk/pub/databases/metagenomics/umgs_analyses/taxonomy_hgr.tab" --cut-dirs 5 -P $LABEL_PATH
wget "ftp://ftp.ebi.ac.uk/pub/databases/metagenomics/umgs_analyses/taxonomy_umgs.tab" --cut-dirs 5 -P $LABEL_PATH
gunzip $DATA_PATH**/*.gz



