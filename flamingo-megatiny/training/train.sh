#!/bin/env bash

export CUDA_VISIBLE_DEVICES=0
NUM_GPU=1

ARGS="
--output_dir ./flamingo-coco
--run_name flamingo-megatiny-vitL-coco
--do_train --do_eval
--optim adamw_torch
--learning_rate 0.0001 
--warmup_steps 500
--lr_scheduler_type constant_with_warmup
--per_device_train_batch_size 16
--per_device_eval_batch_size 16
--gradient_accumulation_steps 1
--evaluation_strategy steps
--eval_steps 500
--num_train_epochs 3
--save_strategy epoch
--save_total_limit 2
--log_level info
--dataloader_num_workers 14
--dataloader_pin_memory True
--fp16
--report_to wandb
--ddp_find_unused_parameters False
"

echo $ARGS

if [ $NUM_GPU == 1 ]; then
    echo "running on a single GPU"
    python ./train.py $ARGS
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node $NUM_GPU ./train.py $ARGS
fi