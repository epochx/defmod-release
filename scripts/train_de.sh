#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=48:00:00
#$ -j y
#$ -cwd
#$ -m ea


source /etc/profile.d/modules.sh              

module load cuda/12.0

source $HOME/miniconda3/bin/activate defmod2


# for MODEL in "facebook/mbart-large-50" "facebook/mbart-large-cc25" "google/mt5-large"
for MODEL in "google/mt5-large"
do

    if [ $MODEL == "google/mt5-large" ]
    then
        BATCH_SIZE=8
    else
        BATCH_SIZE=16
    ficd

    # python ~/defmod/train.py \
    #     --model_name_or_path $MODEL \
    #     --datasets ~/storage-matsuo/data/defmod/monosemic-lemmas-only/de-duden/ \
    #     --output_dir ~/storage-matsuo/results/defmod/de \
    #     --per_device_train_batch_size $BATCH_SIZE \
    #     --per_device_eval_batch_size $BATCH_SIZE \
    #     --val_max_target_length 256 \
    #     --num_train_epochs 20 \
    #     --with_tracking \
    #     --report_to wandb 

    python ~/defmod/train.py \
        --model_name_or_path $MODEL \
        --datasets $HOME/storage-matsuo/data/defmod/monosemic-lemmas-only/en-oxford/ $HOME/storage-matsuo/data/defmod/monosemic-lemmas-only/de-duden/ \
        --output_dir $HOME/storage-matsuo/results/defmod/en-de \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --val_max_target_length 256 \
        --num_train_epochs 20 \
        --with_tracking \
        --report_to wandb

done
