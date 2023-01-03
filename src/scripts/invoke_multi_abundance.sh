#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

### Helper script to invoke abundance evaluation of a model on the mag dataset               ###
### It first runs the prediction on each sample. Afterward, it agregates the outputs.        ###
### This results in 3 tables: Abundance and Normalized Abundance                         		 ###

out_dir=$1
input=$2
config_path=$3
model_path=$4
num_processes=$5 
#num_processes=80
workdir=$PWD
mapping=$6
#mapping=/share/ebuschon/data/hgr_umgs/train_raw/sequence_metadata/species_mapping.tab 
vocab=$7
log=$8
pred=$9
single=${10}
labeled=${11}
multiple=${12}

#vocab=/share/ebuschon/data/k_mer_vocabularies/vocab_12mer.txt

if [ -d ${input}/tmp* ]; 
  then 
  echo "There exist 'tmp' folder in input. Please remove or rename them.";
  exit 1 ;
fi


echo "Creating output directory"
mkdir -p ${out_dir}

echo "Subsetting input data"
cd $input
array=($(ls *.fa))
size=${#array[@]}
working=true
i=0


while $working
do
  mkdir tmp${i}
  ((i_new=$i + $num_processes*2)) 

  if [ $i_new -ge $size ]
    then 
      mv ${array[@]:$i:(($num_processes*2+$size-$i_new))} tmp$i
      working=false
  else
      mv ${array[@]:$i:$num_processes} tmp$i
      ((i+=$num_processes))	
  fi
done



echo "Calculate abundance"
cd $workdir
START=$(date +%s.%N)

python3 multiple_abundance.py \
	--reads-path ${input} \
	--output-path ${out_dir} \
	--class-mapping-path ${mapping} \
	--vocab-path ${vocab} \
	--model-state ${model_path} \
	--config-state ${config_path} \
	--prediction-mode $pred \
	--batch-size 2048 \
	--level-index -1 \
	--n-worker $num_processes\
	--single $single \
	--labeled $labeled\
	--multiple-folder $multiple

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo Model used for inference: $model_path > $log
echo Time for inference: $DIFF
echo Inference time for $input: $DIFF  >> $log

echo "Cleanup"
cd $input
mv tmp*/* .
rmdir tmp*

#add concat

echo "Done"
cd ${workdir}


python3 ../scripts/evaluation/concat_mag.py \
    --in-path ${out_dir} \
    --file-name-pattern abundance.csv \
    --grouping-col Taxon \
    --out-path ${out_dir}/





















