#!/bin/bash

HOSTNAME=`hostname -s`
NODE_NUMBER=`cat -n ${HOSTFILE} | grep $HOSTNAME | awk '{print $1}'`

echo "[${HOSTNAME}] OMPI_COMM_WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}"
echo "[${HOSTNAME}] OMPI_COMM_WORLD_RANK=${OMPI_COMM_WORLD_RANK}"
echo "[${HOSTNAME}] OMPI_COMM_WORLD_LOCAL_SIZE=${OMPI_COMM_WORLD_LOCAL_SIZE}"
echo "[${HOSTNAME}] OMPI_COMM_WORLD_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}"
echo "[${HOSTNAME}] OMPI_COMM_WORLD_NODE_RANK=${OMPI_COMM_WORLD_NODE_RANK}"

export MACHINE_RANK=$((NODE_NUMBER-1))
echo "[${HOSTNAME}] MACHINE_RANK=$MACHINE_RANK"


MODEL="google/mt5-large"
BATCH_SIZE=2

accelerate launch \
    --multi_gpu \
    --mixed_precision "no" \
    --num_processes $NUM_PROCESSES \
    --num_machines $NUM_NODES \
    --same_network \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_backend "static" \
    --dynamo_backend "no" \
    train.py \
    --model_name_or_path $MODEL \
    --datasets $HOME/storage-matsuo/data/defmod/monosemic-lemmas-only/en-oxford/ $HOME/storage-matsuo/data/defmod/monosemic-lemmas-only/de-duden/ \
    --output_dir $HOME/storage-matsuo/results/defmod/en-de \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --val_max_target_length 256 \
    --num_train_epochs 20 \
    --with_tracking \
    --report_to wandb