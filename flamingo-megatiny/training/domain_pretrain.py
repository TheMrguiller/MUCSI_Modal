"""
Use Huggingface Trainer with FlamingoModel.

This is a working demo script which you can adapt to your needs.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
import random

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

from torchvision import transforms as T

import transformers
from transformers import HfArgumentParser, CLIPImageProcessor
from transformers.trainer import Trainer, TrainingArguments
from transformers.optimization import get_constant_schedule_with_warmup

from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor

from eval import evaluate_image_captioning_Bilbao  # don't ask me why this import works
from flamingo_mini.utils import BilbaoCaptions

logger = logging.getLogger(__name__)


class CLIPImageTransform:
    """ experimental. A transform that does apply the transforms of a default CLIPFeatureExtractor """
    vision_processor: CLIPImageProcessor

    def __init__(self, clip_model_type: str):
        self.vision_processor = CLIPImageProcessor.from_pretrained(clip_model_type) # type: ignore

    def __call__(self, image) -> torch.Tensor:
        return self.vision_processor(images=image, return_tensors="pt", padding=True)['pixel_values'] #Wrapper que codifica y preapara la imagen. Se encarga de hacer los patches y luego hace un linear


def prepare_training_dataset_Bilbao(config: FlamingoConfig,dataset_path:List[str]):
    """ prepare a CocoCaptions training dataset """
    transform = T.Compose([ #Con cierta probabilidad da la vuelta a la imagen y procesa la imagen con Clip
        T.RandomHorizontalFlip(),                       
        CLIPImageTransform(config.clip_model_type)
    ])

    def target_transform(captions):
        return f"{random.choice(['', ' '])}<image>{captions}<EOC></s>"

    return BilbaoCaptions(
        dataset=dataset_path,
        transform=transform,
        target_transform=target_transform,
        split_name="train"
    )# Link a la clase de COCO https://github.com/facebookresearch/astmt/blob/master/fblib/dataloaders/coco.py
       

def prepare_evaluation_dataset_Bilbao(config: FlamingoConfig,dataset_path:List[str]):
    return BilbaoCaptions(dataset=dataset_path, 
        transform=CLIPImageTransform(config.clip_model_type),
        split_name="test")


class DataCollator:
    def __init__(self, config: FlamingoConfig):
        self.processor = FlamingoProcessor(config)
        
    def __call__(self, batch):
        pixel_values, sentences = zip(*batch)
        inputs = self.processor(text=sentences)
        pixel_values = torch.stack(pixel_values)
        
        return dict(
            pixel_values=pixel_values,
            labels=inputs['input_ids'],
            **inputs
        )


@dataclass
class FlamingoTrainingArguments(TrainingArguments):
    """ custom arguments """
    eval_coco_captioning_prefix: str = field(default="<image>A picture of")         # It's a common thing to do for COCO image captioning
    eval_coco_captioning_start: int = field(default=0)
    eval_coco_captioning_end: int = field(default=1000)
    

class FlamingoTrainer(Trainer):

    args: FlamingoTrainingArguments
    model: FlamingoModel
    processor: FlamingoProcessor
    eval_dataset: BilbaoCaptions
    
    
    def  evaluate(self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """ override evaluation method to inject custom behavior. 
        TODO this only runs on one GPU, how to do distributed evaluation?
        """
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        metrics = evaluate_image_captioning_Bilbao(self.eval_dataset, self.model, 
            prefix="<image>",
            start=self.args.eval_coco_captioning_start,
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=self.args.dataloader_num_workers
        )
        metrics = {f"{metric_key_prefix}_{k}" : v for k, v in metrics.items()}
        metrics["eval_loss"] =output.metrics["eval_loss"]
        # HF trainer stuff from overridden method
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        
        return metrics
    
    
if __name__ == '__main__':
    parser = HfArgumentParser(FlamingoTrainingArguments)
    training_args: FlamingoTrainingArguments
    training_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format=f'%(asctime)s {training_args.run_name} %(message)s', 
        datefmt='%H:%M:%S',
        #force=True,
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler(f'{args.output_dir}/out.log')
        ]    
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    #datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.info(str(training_args))

    logger.info('loading model...')
 
    
    ## Pretained model
    model = FlamingoModel.from_pretrained('landersanmi/flamingo-megatiny-opt')
    config=model.config
    print("###################################################")
    print(f"MOdel config:{config}")
    print(model.device)

    # model = FlamingoModel(config) # Learning from scratch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model.device)
    model.train()

    #################################################################
    # datasets
    #################################################################
    path = ["TheMrguiller/BilbaoCaptions","landersanmi/BilbaoCaptions2"]
    #path = ["landersanmi/BilbaoCaptions2"]
    logger.info('loading datasets...')
    train_dataset = prepare_training_dataset_Bilbao(config, path)
    eval_dataset = prepare_evaluation_dataset_Bilbao(config, path)    
    #################################################################
    # optimizer, scheduler, trainer
    #################################################################
    # optimizer = AdamW(model.parameters_trainable(), training_args.learning_rate)
    # scheduler = get_constant_schedule_with_warmup(optimizer, training_args.warmup_steps)

    trainer = FlamingoTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(config),
        # optimizers=(optimizer, scheduler)
    )

    #################################################################
    # training loop
    #################################################################
    logger.info('start training.')

    if training_args.resume_from_checkpoint is not None:
        trainer.train(training_args.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.evaluate(eval_dataset)

