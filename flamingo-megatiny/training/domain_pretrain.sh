#!/bin/env bash

export CUDA_VISIBLE_DEVICES=0
NUM_GPU=1

ARGS="
--output_dir ./flamingo-coco-opt-domain
--run_name flamingo-megatiny-vitL-coco-opt-domain
--do_train --do_eval
--optim adamw_torch
--learning_rate 0.0001 
--warmup_steps 1000
--lr_scheduler_type constant_with_warmup
--per_device_train_batch_size 4
--per_device_eval_batch_size 4
--gradient_accumulation_steps 1
--evaluation_strategy steps
--eval_steps 1936
--num_train_epochs 10
--save_strategy epoch
--save_total_limit 15
--log_level info
--dataloader_num_workers 4
--dataloader_pin_memory True
--fp16
--report_to wandb
--ddp_find_unused_parameters False
"

echo $ARGS

if [ $NUM_GPU == 1 ]; then
    echo "running on a single GPU"
    python ./domain_pretrain.py $ARGS
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node $NUM_GPU ./domain_pretrain.py $ARGS
fi