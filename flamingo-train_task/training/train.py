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
from torchvision.datasets import CocoCaptions

import transformers
from transformers import HfArgumentParser, CLIPImageProcessor
from transformers.trainer import Trainer, TrainingArguments
from transformers.optimization import get_constant_schedule_with_warmup

from flamingo_mini_task import FlamingoConfig, FlamingoModel, FlamingoProcessor

from eval import evaluate_image_captioning  # don't ask me why this import works
from flamingo_mini_task.utils import BilbaoCaptions,BilbaoQA,VQAv2

logger = logging.getLogger(__name__)


# get images and annotations from https://cocodataset.org/#download
COCO_ROOT      = '/nfs/data3/zhangya/coco2017/images'
COCO_ANN_TRAIN = '/nfs/data3/hansmair/coco2017/captions_train2017.json'
COCO_ANN_VAL   = '/nfs/data3/hansmair/coco2017/captions_val2017.json'

# get images and annotations from https://visualqa.org/download.html
VQAV2_ROOT = 'VQAV2/val2014'
VQAV2_ANN_QUEST_VAL = 'VQAV2/v2_OpenEnded_mscoco_val2014_questions.json'
VQAV2_ANN_VAL = 'VQAV2/v2_mscoco_val2014_annotations.json'


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

    def target_transform(data):
        #Depending on the task we change the task token
        if data["answer"]=="":
            return f"{random.choice(['', ' '])}[COT][CONTEXT]<image>{data['question']}{data['choices']}[ANSWER]"
        else:
            return f"{random.choice(['', ' '])}[QA][CONTEXT]<image>{data['question']}{data['choices']}[ANSWER]"


    return BilbaoQA(
        dataset=dataset_path,
        transform=transform,
        target_transform=target_transform,
        split_name="train"
    )# Link a la clase de COCO https://github.com/facebookresearch/astmt/blob/master/fblib/dataloaders/coco.py
       

def prepare_evaluation_dataset_Bilbao(config: FlamingoConfig,dataset_path:List[str]):
    return BilbaoQA(dataset=dataset_path, 
        transform=CLIPImageTransform(config.clip_model_type),
        
        split_name="test")
def prepare_evaluation_dataset_BilbaoQA(config: FlamingoConfig,dataset_path:List[str],split_name="train"):
    def target_transform(data):
        #Depending on the task we change the task token
        if data["answer"]=="":
            return f"{random.choice(['', ' '])}[COT][CONTEXT]<image>{data['question']}{data['choices']}[ANSWER]"
        else:
            return f"{random.choice(['', ' '])}[QA][CONTEXT]<image>{data['question']}{data['choices']}[ANSWER]"


    return BilbaoQA(dataset=dataset_path, 
        transform=CLIPImageTransform(config.clip_model_type),
        target_transform=target_transform,
        split_name=split_name)

       
def prepare_evaluation_dataset_VQAv2(config: FlamingoConfig):
    
    transform = T.Compose([ #Con cierta probabilidad da la vuelta a la imagen y procesa la imagen con Clip
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),                       
        CLIPImageTransform(config.clip_model_type)
    ])
    
    def target_transform(data):
            print(data)
            return f"{random.choice(['', ' '])}[QA][CONTEXT]<image>{random.choice(data)}[ANSWER]"
    
    return VQAv2(
        image_folder=VQAV2_ROOT,
        questions_file=VQAV2_ANN_QUEST_VAL,
        annotations_file=VQAV2_ANN_VAL,
        transform=transform,
        target_transform=target_transform,
    )


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
class DataCollatorQA:
    def __init__(self, config: FlamingoConfig):
        self.processor = FlamingoProcessor(config)
        
    def __call__(self, batch):
        pixel_values, sentences, labels = zip(*batch)
        inputs = self.processor(text=sentences)
        # max_length_1 = max(len(sequence) for sequence in inputs['input_ids'])
        # print(f"Input max length:{max_length_1}")
        label=self.processor(text=labels)
        # max_length_2 = max(len(sequence) for sequence in label['input_ids'])
        # print(f"Label max length:{max_length_2}")
        pixel_values = torch.stack(pixel_values)
        # print("/////////////////////////////////")
        # print(f"length input:{len(sentences)},length labels:{len(labels)}")
        # print(f"input:{inputs['input_ids'].shape},label:{label['input_ids'].shape}")
        return dict(
            pixel_values=pixel_values,
            labels=label['input_ids'],
            **label
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
    #eval_dataset: BilbaoQA
    eval_dataset: VQAv2
    @torch.no_grad()
    def evaluate(self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """ override evaluation method to inject custom behavior. 
        TODO this only runs on one GPU, how to do distributed evaluation?
        """
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        # start_time = time.time()

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
        print(output.metrics["eval_loss"])
        metrics = evaluate_image_captioning(self.eval_dataset, self.model, 
            prefix="",
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
# def compute_metrics(eval_pred):
#     logits,labels=eval_pred
#     processor = FlamingoProcessor(model.config)
#     captions = processor.tokenizer.batch_decode(
#             logits, skip_special_tokens=True)
#     captions = [processor.remove_tags(t) for t in captions]
#     print(captions)
    
#     return {}

    
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
            logging.StreamHandler()
            # logging.FileHandler(f'{args.output_dir}/out.log')
        ]    
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    #datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.info(str(training_args))

    logger.info('loading model...')
    # config = FlamingoConfig(
    #     clip_model_type='openai/clip-vit-large-patch14',
    #     lm='facebook/opt-125m',
    #     dim=768,
    #     dim_visual=1024,
    #     xattn_act='sqrelu',
    #     resampler_act='sqrelu'
    # )
    # model = FlamingoModel(config)

    #model = FlamingoModel.from_pretrained('landersanmi/flamingo-megatiny-opt',ignore_mismatched_sizes=True)
    model = FlamingoModel.from_pretrained('/home/lander/Documentos/GitHub/MUCSI_Modal/flamingo-train_task/training/flamingo-megatiny-opt-QA/checkpoint-20620',ignore_mismatched_sizes=True)
    config=model.config
    # model.lm.
    print(f"Model config:{config}")
    device=device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("%%%%%%%%%%%%%%%%%%%%%%")
    print(model.device)
    model.train()

    #################################################################
    # datasets
    #################################################################
    path = ["TheMrguiller/ScienceQA"]#,"TheMrguiller/BilbaoQA","TheMrguiller/BilbaoQA2"]
    # path = ["TheMrguiller/ScienceQA","TheMrguiller/BilbaoQA","TheMrguiller/BilbaoQA2"]
    logger.info('loading datasets...')
    train_dataset = prepare_training_dataset_Bilbao(config,path)
    #eval_dataset = prepare_evaluation_dataset_BilbaoQA(config,path,split_name="test")
    eval_dataset = prepare_evaluation_dataset_VQAv2(config)
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
        data_collator=DataCollatorQA(config),
        #compute_metrics=compute_metrics,
        # optimizers=(optimizer, scheduler)
    )

    #################################################################
    # training loop
    #################################################################
    logger.info('start training.')
    trainer.evaluate(eval_dataset)

    #if training_args.resume_from_checkpoint is not None:
        #trainer.train(training_args.resume_from_checkpoint)
    #else:
        #trainer.train()
    
