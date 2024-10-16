#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd
#$ -m ea


source /etc/profile.d/modules.sh              

module load cuda/11.7
module load nccl/2.13/2.13.4-1

source $HOME/miniconda3/bin/activate defmod

cd defmod

# "google/mt5-large" "facebook/mbart-large-50"
for MODEL in "facebook/mbart-large-cc25" 
do
    if [ $MODEL == "google/mt5-large" ]
    then
        BATCH_SIZE=8
    else
        BATCH_SIZE=16
    fi

    python train.py \
        --model_name_or_path  $MODEL \
        --datasets ~/data/defmod/monosemic-lemmas-only/en-oxford/ \
        --output_dir ~/results/defmod/en \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --val_max_target_length 256 \
        --num_train_epochs 20 \
        --with_tracking \
        --report_to wandb
done
