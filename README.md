# MetaTransformer
Metagenomic classification using Transformer

**Author**: Etienne Buschong and Alexander Wichmann

## Table of contents
1. [Folder structure](#1-folder-structure)
2. [Setup](#2-setup)
3. [Model training](#3-model-training)
4. [Configuration file](#4-configuration-file)
5. [Testing the model](#5-testing-the-model)
6. [Training and testing data preparation](#6-training-and-testing-data-preparation)
7. [Additional data](#7-additional-data)

## 1. Folder structure 
The project is structured into the following folders:
- **bin**: Contains third party scripts that are used by other scripts in the projects
- **scripts**: Contains scripts for training data generation and plotting of results graphics
- **src**: Contains the actual code for model definition, training and testing
- **sequence_metadata**: Contains genus and species mapping as well as taxonomy information and weightings for the model (see configuration file) 

## 2. Setup
To use the code you need to have Python3 and CUDA 11.1 installed. Preferably, use a virtual environment to install the dependencies. To install the dependencies invoke 
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
on the root level of the project.

To use the locality-sensitive hashing input transformation you need to compile the Cython extension. To do this invoke
```
python3 setup.py build_ext --inplace
```
in the `src` folder.

For training as well as inference embedding tables are needed. They are retrievable using https://doi.org/10.5281/zenodo.7594864 for different k-mer sizes.

If you want to use our pretrained model, both genus and species models are available under:

genus: https://doi.org/10.5281/zenodo.7594286
species: https://doi.org/10.5281/zenodo.7594005

## 3. Model training
For model training invoke the `train.py` script in the following way
```
python3 train.py experiment_name=<name-of-training> \
                 experiment_base_dir=<path-where-to-save-training-data> \
                 cfg_path=<path-to-config-yaml-file> \
                 data_path_root=<path-prefix-to-training-data> 
``` 
The last parameter `data_path_root` acts as a path-prefix to enable training on multiple machines where the training data is located at another path (i.e. training on a local machine and a cluster).

The script will create a training folder with the following contents:
- config.yaml: a copy of the configuration that is used for this model training
- tensorboard: contains tensorboard logs from the training
- checkpoints: contains the model checkpoints
- train.log: logs from the training (progress, metrics, errors etc.)

To resume the training of a model invoke the `train.py` script in the following way
```bash
python3 train.py resume_dir=<path-to-training-data-folder>
```
The script will automatically resume the training from the last checkpoint. 

## 4. Configuration file
Before training a model, you can adapt the configuration to your needs in the config files located under the config folder at `src/config`.
The main config file is `config.yaml`. 
The other config files in the different subdirectories denote different configuration choices like model type, dataset etc. which can be changed in the `config.yaml` file.

The most important parameters in the `config.yaml` file are:
- dataset: choose your dataset to train on. Choices are `species_hgr_umgs` or `genus_hgr_umgs`
- model: choose your model to train. Choices are `embed_lstm_attention` or `classification_transformer`
- input_module: choose the type of tokenization and input encoding. Choices are `vocab` (k-mer encoding), `lsh`, `hash_embedding`, `one_hot_embed` and `bpe`.

All parameters are commented so adjust them to your needs.

**Note**: Not all parameters are used all the time. For example `num_buckets` or `num_hash_functions` are only used when either `lsh` or `hash_embedding` is activated as the input_module.

## 5. Testing the model
To test the models on the benchmark datasets or inference the pretrained models with ur data, you can either invoke the test scripts directly or use the helper scripts located under `src/scripts`.

We provide two test scripts. 

`invoke_multi_abundance.sh` is used to calculate the abundance for different sceanrios

To invoke it use the following command
```
./src/scripts/invoke_multi_abundance.sh \
    <output directory>                  \
    <input directory>                   \
    <path to config of model>           \
    <path to model>                     \
    <number threads>                    \
    <path to mapping file>              \
    <path to vocab file>                \
    <path to log file>                  \
    <level-index>                       \
    <if each sample is a single file>   \   #True/False
    <if input file is labeled>          \   #True/False
    <if the input consists of multiple folders>    #True/False
```
The argument `level-index` is used when a multi-level classifier needs to be tested. Since those models predict on multiple ranks we have to indicate which rank we want to use for prediction. The value defaults to -1 when a single-level model is used.


**Note 1** When manually invoking the benchmark scripts and having multiple GPUs, you may need to use `export CUDA_VISIBLE_DEVICES=<number-of-device>` to select the correct GPU.

**Note 2** When using more than 1 worker for data loading you need to manually split the files into chunks. You need at least one chunk per worker.

**Note 3** If the input data is too large, you can also split it into multiple files.

To split the input into multiple files you can use external software like BBtools or our simple script "src/script/subset_fasta.py" with the parameters:
```
python src/script/subset_fasta.py \
  --input <path to input file>  \
  --n-reads <number of reads to split the file into> \
  --out-folder <path to out-folder> 
```

## 6. Training and testing data preparation
### 6.1 Training data
You can download the training data from
```
http://ftp.ebi.ac.uk/pub/databases/metagenomics/umgs_analyses/  (Browser)
ftp:://ftp.ebi.ac.uk/pub/databases/metagenomics/umgs_analyses/  (Ftp)
```
You need to download the genomes (umgs and hgr) as well as both taxonomy files.

Then use the script `label_train_data.py` located under `scripts/deep_microbes_scripts` to label the train data as follows
```
python3 label_train_data.py --train-path <path-to-training-genomes> \
                            --tax-path <path-to-folder-containing-both-taxonomy-files> \
                            --multi-level <flag indicating whether to label on both genus and species level>
```
It will create labeled sequences on both genus and species level and also creates class mapping files (i.e. class to class index mappings).

To create read data from your training genomes use the `generate_training_reads.sh` script located at `scripts/read_transform` in the following way

```
./generate_training_reads.sh -i <path-to-input-genome-folder> \
                             -c <coverage to achieve on the longest genome> \
                             -o <path-to-output-folder> \
                             -n <number of tasks to use during read generations> 
```
After labeling the reads can be modified or directly used for training a custom model.

### 6.2 Testing data
The benchmark gut dataset can be downloaded [here](https://mail2sysueducn-my.sharepoint.com/personal/liangqx7_mail2_sysu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fliangqx7%5Fmail2%5Fsysu%5Fedu%5Fcn%2FDocuments%2FBenchmark%5Fdatasets%2Fgut%5FMAG%2FMAG%5Freads%5F150bp%2Etar%2Egz&parent=%2Fpersonal%2Fliangqx7%5Fmail2%5Fsysu%5Fedu%5Fcn%2FDocuments%2FBenchmark%5Fdatasets%2Fgut%5FMAG&ga=1).

The benchmark gut metadata can be downloaded [here](https://raw.githubusercontent.com/MicrobeLab/DeepMicrobes-data/master/benchmark_datasets/metadata_gut-derived_MAGs.txt)

The benchmark mock dataset can be downloaded [here](https://mail2sysueducn-my.sharepoint.com/personal/liangqx7_mail2_sysu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fliangqx7%5Fmail2%5Fsysu%5Fedu%5Fcn%2FDocuments%2FBenchmark%5Fdatasets%2Fmock%5Freal&ga=1)

The benchmark mock metadata can be downloaded [here](https://github.com/MicrobeLab/DeepMicrobes-data/blob/master/benchmark_datasets/mock_ground_truth_templete.csv)

The mock communities are given as FASTQ files. To convert them to FASTA format use the `fq2fa.py` script located under `scripts/read_transform` in the following way 
```
python3 fq2fa.py -p <path-to-directory-with-fastq-files>
```

To label the gut benchmark data use the script `label_gut_test_data.py` located under `scripts/deep_microbes_scripts` in the following way
```
python3 label_gut_test_data.py -p <path-to-gut-genomes> \
                               -t <path-to-class-mapping> \
                               -m <path-to-gut-metadata-file> \
                               -o <output-path> \
                               -lv <species or genus>
```

To label the mock benchmark data use the script `label_mock_test_data.py` located under `scripts/deep_microbes_scripts` in the following way
```
python3 label_mock_test_data.py -p <path-to-mock-community> \
                                -t <path-to-class-mapping> \
                                -m <path-to-mock-metadata-file> \
                                -o <output-path>
```


Before prediction, you may need to interleave paired-end reads, based on the prediction mode (1/2/4). The prediction mode indicates how many consecutive reads from a read file are averaged for prediction.

To interleave paired-end reads invoke the `interleave_reads_paired.sh` script located under `scripts/read_transform` in the following way

```
./interleave_reads_paired.sh -f <path-to-forward-strand-reads> \
                             -r <path-to-reverse-strand-reads> \
                             -o <path-where-to-save-interleaved-reads>
``` 

In case that you also want to create the reverse-complement of both the forward and reverse strand invoke the `interleave_reads_bi_paired.sh` script located at `scripts/read_transform` in the following way

```
./interleave_reads_bi_paired.sh --forward-path <path-to-forward-strand-reads> \
                                --reverse-path <path-to-reverse-strand-reads> \
                                --output-path <path-where-to-save-interleaved-reads>
``` 

## 7. Additional data

Additional data can be accessed via https://github.com/alexwichma/MetaTransformer_data.

