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

python zero_shot.py \
    --model ~/storage-matsuo/pretrained/llama2-HF/Llama-2-13b-chat-hf/ \
    --load_in_4bit \
    --data ~/storage-kirt/data/defmod/monosemic-lemmas-only/fr-larousse/ \
    --output_dir ~/storage-matsuo/results/defmod/fr \
    --num_shots 3 \
    --batch_size 32 \
    --seed 22

python zero_shot.py \
    --model ~/storage-matsuo/pretrained/llama2-HF/Llama-2-13b-chat-hf/ \
    --load_in_4bit \
    --data ~/storage-kirt/data/defmod/monosemic-lemmas-only/fr-larousse/ \
    --output_dir ~/storage-matsuo/results/defmod/fr \
    --num_shots 5 \
    --batch_size 32 \
    --seed 2


# for NUM_SHOTS in 5
# do
#     for SEED in 2
#     do
        # python zero_shot.py \
        #     --model ~/storage-matsuo/pretrained/llama2-HF/Llama-2-7b-chat-hf/ \
        #     --load_in_4bit \
        #     --data ~/data/defmod/monosemic-lemmas-only/fr-larousse/ \
        #     --output_dir ~/results/defmod/fr \
        #     --num_shots $NUM_SHOTS \
        #     --batch_size 32 \
        #     --seed $SEED
    
        # python zero_shot.py \
        #     --model ~/storage-matsuo/pretrained/llama2-HF/Llama-2-13b-chat-hf/ \
        #     --load_in_4bit \
        #     --data ~/storage-kirt/data/defmod/monosemic-lemmas-only/fr-larousse/ \
        #     --output_dir ~/storage-matsuo/results/defmod/fr \
        #     --num_shots $NUM_SHOTS \
        #     --batch_size 32 \
        #     --seed $SEED
    
        # python zero_shot.py \
        #     --model ~/storage-matsuo/pretrained/mistralai/Mistral-7B-Instruct-v0.1// \
        #     --use_default_system_prompt \
        #     --load_in_4bit \
        #     --data ~/storage-kirt/data/defmod/monosemic-lemmas-only/fr-larousse/ \
        #     --output_dir ~/storage-matsuo/results/defmod/fr \
        #     --num_shots $NUM_SHOTS \
        #     --batch_size 32 \
        #     --seed $SEED
            
#     done    
# done

