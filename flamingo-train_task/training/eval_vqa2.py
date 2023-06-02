import matplotlib.pyplot as plt
import torch
from flamingo_mini_task.utils import load_url,BilbaoQA
from flamingo_mini_task import FlamingoModel, FlamingoProcessor
from datasets import load_dataset,concatenate_datasets
from train import prepare_evaluation_dataset_BilbaoQA,DataCollatorQA,FlamingoTrainer,FlamingoTrainingArguments
from transformers.trainer import Trainer, TrainingArguments
from typing import Dict, Iterable, List, Optional, Tuple
from torch.utils.data import Dataset
from eval import evaluate_image_captioning

batch_size=16
num_workers=32
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class FlamingoTrainer(Trainer):

    args: FlamingoTrainingArguments
    model: FlamingoModel
    processor: FlamingoProcessor
    eval_dataset: BilbaoQA
    # eval_dataset: VQAv2
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
        metrics = evaluate_image_captioning(self.eval_dataset, self.model, 
            prefix="",
            start=0,
            batch_size=batch_size,
            num_workers=num_workers
        )
        metrics = {f"{metric_key_prefix}_{k}" : v for k, v in metrics.items()}
        self._memory_tracker.stop_and_update_metrics(metrics)
        print(metrics)
        return metrics
if __name__ == '__main__':
    model = FlamingoModel.from_pretrained('dhansmair/flamingo-tiny',ignore_mismatched_sizes=True)
    device=device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config=model.config
    model_list = ["TheMrguiller/Flamingo-tiny_ScienceQA_COT-QA","TheMrguiller/Flamingo-mini-Bilbao_Captions-task_ScienceQA",
                  "TheMrguiller/Flamingo-tiny-task_BilbaoQA-ScienceQA","TheMrguiller/Flamingo-tiny-Bilbao_Captions-task_ScienceQA",
                  "TheMrguiller/Flamingo-mini-task_ScienceQA_BilbaoQA","TheMrguiller/Flamingo-mini-Bilbao_Captions-task_BilbaoQA-ScienceQA0",
                  "TheMrguiller/Flamingo-tiny-Bilbao_Captions-task_BilbaoQA-ScienceQA"] #model list to evaluate
    path = ["landersanmi/VQAv2"]#,"TheMrguiller/BilbaoQA","TheMrguiller/BilbaoQA2"]
    
    eval_dataset = prepare_evaluation_dataset_BilbaoQA(config,path,split_name="train")
    for model_name in model_list:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"Model name:{model_name}")
        model = FlamingoModel.from_pretrained(model_name,ignore_mismatched_sizes=True)
        model.to(device)
        model.eval()
        trainer = FlamingoTrainer(
            model,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorQA(config),
            )
        trainer.evaluate(eval_dataset)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
     
    
    
    
    
    
    

    

    