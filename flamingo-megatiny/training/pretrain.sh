#!/bin/env bash
export LANG=en_US.UTF-8  # For solving Meteor crash
export CUDA_VISIBLE_DEVICES=0
NUM_GPU=1

ARGS="
--output_dir ./flamingo-coco-opt-vizwiz-evalcoco2
--run_name flamingo-vitL-coco-opt-vizwiz-se-evalcoco2
--do_train --do_eval
--optim adamw_torch
--learning_rate 0.0001 
--warmup_steps 1000
--lr_scheduler_type constant_with_warmup
--per_device_train_batch_size 8
--per_device_eval_batch_size 32
--gradient_accumulation_steps 1
--evaluation_strategy steps
--eval_steps 500
--num_train_epochs 15
--save_strategy epoch
--save_total_limit 15
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
    python ./pretrain.py $ARGS
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node $NUM_GPU ./pretrain.py $ARGS
fi