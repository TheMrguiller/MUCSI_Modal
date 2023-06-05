# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import math
import sys
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.utils import ModelEma
from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets_2 import get_sentencepiece_model_for_beit3

import utils


class TaskHandler(object):
    def __init__(self) -> None:
        self.metric_logger = None
        self.split = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

    def eval_batch(self, model, **kwargs):
        raise NotImplementedError()

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split

    def after_eval(self, **kwargs):
        raise NotImplementedError()

class VQAHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.predictions = []
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.label2ans = None

    def train_batch(self, model, image, language_tokens, padding_mask, labels):
        logits = model(
            image=image, question=language_tokens, 
            padding_mask=padding_mask)
        return {
            "loss": self.criterion(input=logits.float(), target=labels.float()) * labels.shape[1], 
        }

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.predictions.clear()
        self.metric_logger = metric_logger
        self.label2ans = data_loader.dataset.label2ans

    def eval_batch(self, model, image, language_tokens, padding_mask, labels=None, qid=None):
        logits = model(
            image=image, question=language_tokens, 
            padding_mask=padding_mask)
        batch_size = language_tokens.shape[0]
        if labels is not None:
            scores = utils.VQAScore()(logits, labels) * 100.0
            self.metric_logger.meters['score'].update(scores.item(), n=batch_size)
        else:
            _, preds = logits.max(-1)
            for image_id, pred in zip(qid, preds):
                self.predictions.append({
                    "question_id": image_id.item(), 
                    "answer": self.label2ans[pred.item()], 
                })

    def after_eval(self, **kwargs):
        if len(self.predictions) == 0:
            print('* Score {score.global_avg:.3f}'.format(score=self.metric_logger.score))
            return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "score"
        else:
            return self.predictions, "prediction"


def get_handler(args):
    if args.task == "vqav2" or args.task == "bilbao":
        return VQAHandler()
    else:
        raise NotImplementedError("Sorry, %s is not support." % args.task)


def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable, 
        optimizer: torch.optim.Optimizer, device: torch.device, 
        handler: TaskHandler, epoch: int, start_steps: int, 
        lr_schedule_values: list, loss_scaler, max_norm: float = 0, 
        update_freq: int = 1, model_ema: Optional[ModelEma] = None, 
        log_writer: Optional[utils.TensorboardLogger] = None, 
        task = None, mixup_fn=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if epoch == 0 and data_iter_step >= 5000:
            return
        
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
                    #param_group["lr"] = lr_schedule_values[global_step]
        # put input data into cuda
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            # print("input %s = %s" % (tensor_key, data[tensor_key]))
            if loss_scaler is None and tensor_key.startswith("image"):
                data[tensor_key] = data[tensor_key].half()

        # mixup for imagenet finetuning
        if mixup_fn is not None:
            data["image"], data["label"] = mixup_fn(data["image"], data["label"])
        
        if task in ["coco_captioning", "nocaps"]:
            data["global_step"] = global_step

        if loss_scaler is None:
            results = handler.train_batch(model, **data)
        else:
            with torch.cuda.amp.autocast():
                results = handler.train_batch(model, **data)

        loss = results.pop("loss")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                    
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs = {
                "loss": loss_value, 
            }
            for key in results:
                kwargs[key] = results[key]
            log_writer.update(head="train", **kwargs)

            kwargs = {
                "loss_scale": loss_scale_value, 
                "lr": max_lr, 
                "min_lr": min_lr, 
                "weight_decay": weight_decay_value, 
                "grad_norm": grad_norm, 
            }
            log_writer.update(head="opt", **kwargs)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, handler):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for data in metric_logger.log_every(data_loader, 10, header):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            handler.eval_batch(model=model, **data)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return handler.after_eval()
