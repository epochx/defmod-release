#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd
#$ -m ea


source /etc/profile.d/modules.sh              

module load cuda/11.7
module load nccl/2.13/2.13.4-1

source $HOME/miniconda3/bin/activate defmod

cd defmod

# for NUM_SHOTS in 0  -> batch_size 16
for NUM_SHOTS in 0
do
    for SEED in 2 22 1729
    do
      
        python zero_shot.py \
            --model ~/storage-matsuo/pretrained/llama2-HF/Llama-2-13b-chat-hf/ \
            --data ~/storage-kirt/data/defmod/monosemic-lemmas-only/en-oxford/ \
            --output_dir ~/storage-matsuo/results/defmod/en \
            --num_shots $NUM_SHOTS \
            --batch_size 32 \
            --seed $SEED \
            --device_map balanced_low_0 \
            --use_bfloat16
        
        python zero_shot.py \
            --model ~/storage-matsuo/pretrained/mistralai/Mistral-7B-Instruct-v0.1/ \
            --use_default_system_prompt \       
            --data ~/storage-kirt/data/defmod/monosemic-lemmas-only/en-oxford/ \
            --output_dir ~/storage-matsuo/results/defmod/en \
            --num_shots $NUM_SHOTS \
            --batch_size 32 \
            --seed $SEED \
            --device_map balanced_low_0 \
            --use_bfloat16

    done    
done




