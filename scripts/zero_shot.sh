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

for NUM_SHOTS in 0
do

    # python zero_shot.py \
    #     --model ~/storage-matsuo/pretrained/llama2-HF/Llama-2-7b-chat-hf/ \
    #     --load_in_4bit \
    #     --data ~/data/defmod/monosemic-lemmas-only/es-drae/ \
    #     --output_dir ~/results/defmod/es \
    #     --num_shots $NUM_SHOTS \
    #     --batch_size 0
   
    python zero_shot.py \
        --model ~/storage-matsuo/pretrained/mistralai/Mistral-7B-Instruct-v0.1/ \
        --use_default_system_prompt \
        --data ~/data/defmod/monosemic-lemmas-only/es-drae/ \
        --output_dir ~/results/defmod/es \
        --num_shots $NUM_SHOTS \
        --batch_size 8

    python zero_shot.py \
        --model ~/storage-matsuo/pretrained/mistralai/Mistral-7B-Instruct-v0.1/ \
        --use_default_system_prompt \
        --data ~/data/defmod/monosemic-lemmas-only/fr-larousse/ \
        --output_dir ~/results/defmod/fr \
        --num_shots $NUM_SHOTS \
        --batch_size 8

    python zero_shot.py \
        --model ~/storage-matsuo/pretrained/mistralai/Mistral-7B-Instruct-v0.1/ \
        --use_default_system_prompt \
        --data ~/data/defmod/monosemic-lemmas-only/pt-dicio/ \
        --output_dir ~/results/defmod/pt \
        --num_shots $NUM_SHOTS \
        --batch_size 8

 
done

