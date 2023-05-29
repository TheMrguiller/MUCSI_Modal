from typing import Optional, List, Dict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import CocoCaptions
# from pycocoevalcap.eval import COCOEvalCap
from flamingo_mini_task.utils import BilbaoCaptions,BilbaoQA
from flamingo_mini_task import FlamingoModel, FlamingoProcessor
import evaluate
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
class MyDatasetWrapper(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target,label = self.dataset[index]
        image_id = index
        # print(image.shape)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(image_id)
        return image_id, image, target, label


# @torch.no_grad()
# def evaluate_image_captioning( #https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py
#     dataset: BilbaoCaptions,
#     model: FlamingoModel, 
#     *,
#     prefix: str = "<image>",
#     start = 0,
#     end: Optional[int] = None,
#     verbose: bool = True,
#     batch_size: int = 64,
#     num_workers: int = 8, 
# ) -> Dict[str, float]:

#     processor = FlamingoProcessor(model.config)
#     results: List[dict] = []

#     wrapper = MyDatasetWrapper(dataset)
#     wrapper = Subset(wrapper, range(start, end if end is not None else len(wrapper)))
#     loader = DataLoader(
#         wrapper, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
#         num_workers=num_workers)

#     for image_ids, pixels in tqdm(loader, disable=not verbose):
#         captions = model.generate_captions(
#             processor, 
#             pixel_values=pixels.to(model.device),
#             prompt=prefix
#         )
        
#         for image_id, caption in zip(image_ids.tolist(), captions):
#             results.append(dict(image_id=image_id, caption=caption))

#     coco_result = dataset.coco.loadRes(results)
#     coco_eval = COCOEvalCap(dataset.coco, coco_result)
#     coco_eval.params['image_id'] = coco_result.getImgIds()
#     coco_eval.evaluate()
#     return coco_eval.eval
def calculate_accuracy(hipotesis,reference):
    if hipotesis == reference:
        return 1
    else:
        return 0
@torch.no_grad()
def evaluate_image_captioning( #https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py
    dataset: BilbaoQA,
    model: FlamingoModel, 
    *,
    prefix: str = "",
    start = 0,
    end: Optional[int] = None,
    verbose: bool = True,
    batch_size: int = 64,
    num_workers: int = 8, 
) -> Dict[str, float]:

    processor = FlamingoProcessor(model.config)
    accuracy_sum=0
    total_QA=0
    captions_COT = []
    ref_captions_COT =[]
    wrapper = MyDatasetWrapper(dataset)
    
    wrapper = Subset(wrapper, range(start, end if end is not None else len(wrapper)))
    kwargs = {'num_workers': 1, 'pin_memory': True} 
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_loader = torch.utils.data.DataLoader(wrapper, batch_size=batch_size,shuffle=False,drop_last=False, **kwargs)
    
    # print(train_loader.device)
    loader = DataLoader(
        wrapper, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=num_workers, persistent_workers=True)
    # tokenizer = PTBTokenizer()
    # gts = {} #dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
    # res = {} #dictionary with key <image> and value <tokenized reference sentence>
    
                
    # print(loader.dataset[0])
    
    for image_ids, pixels ,targets, labels in tqdm(loader):
        
        for image_id,pixel,target,label in zip(image_ids.tolist(),pixels,targets,labels):
            caption = model.generate_captions(
            processor, 
            pixel_values=pixel.to(model.device),
            prompt=target,
            device=device
            )

            pixel.to("cpu")
            
            label = label.replace("<image>","")
            label = label.replace("<EOC></s>","")
            caption=caption[0].split("[ANSWER]")[1].strip()
            label=label.split("[ANSWER]")[1]

            print("--------------------- TARGET -----------------------")
            print(target)
            print("--------------------- CAPTION -----------------------")

            print(caption)
            print("--------------------- LABEL -----------------------")
            print(label)

            if "[QA]" in target:
                #calculate accuracy
            
                accuracy_sum += calculate_accuracy(caption,label)
                total_QA += 1
            if "[COT]" in target:
                caption = caption[0].split("[ANSWER]")[1]
                label=label.split("[ANSWER]")[1]

                captions_COT.append(caption)
                ref_captions_COT.append(label)
    


            # print(caption)
            # print(image_id)
            # gts[image_id]= {"caption":caption}
            # res[image_id]= {"caption":ref_caption}
        # if image_ids >0:
        #     break
            
    # gts  = tokenizer.tokenize(gts)
    # res = tokenizer.tokenize(res)
    # cider_metric=Cider()
    # spice_metric=Spice()
    # cider_result,_=cider_metric.compute_score(gts=gts,res=res)
    # print(f"CIDeR:{cider_result}")
    # spice_result,_=spice_metric.compute_score(gts=gts,res=res)

    #Evaluate based in meteor,rouge.Novel metrics cider y spider

    # bleu_metric = evaluate.load("bleu")
    # bleu_result = bleu_metric.compute(predictions=captions_COT, references=ref_captions_COT)
    # meteor_metric = evaluate.load('meteor')
    # meteor_result=meteor_metric.compute(predictions=captions_COT, references=ref_captions_COT)
    # rougue_metric=evaluate.load('rouge')
    # rouge_result=rougue_metric.compute(predictions=captions_COT, references=ref_captions_COT)

    # bleu_metric = evaluate.load("bleu")
    # try:
    #     bleu_result = bleu_metric.compute(predictions=captions_COT, references=ref_captions_COT)
    # except:
    #     bleu_result = {'bleu':0}

    # meteor_metric = evaluate.load('meteor')
    # meteor_result=meteor_metric.compute(predictions=captions_COT, references=ref_captions_COT)
    # rougue_metric=evaluate.load('rouge')
    # rouge_result=rougue_metric.compute(predictions=captions_COT, references=ref_captions_COT)

    accuracy_score = accuracy_sum/total_QA
    # coco_result = dataset.coco.loadRes(results)
    # coco_eval = COCOEvalCap(dataset.coco, coco_result)
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # coco_eval.evaluate()
    # result = {"Bleu":bleu_result["bleu"],"Meteor":meteor_result["meteor"],"Rouge":rouge_result["rougeL"],"CIDEr":cider_result,"SPICE":spice_result}
    # result = {"Bleu":bleu_result["bleu"],"Meteor":meteor_result["meteor"],"Rouge":rouge_result["rougeL"],"CIDEr":cider_result}
    # result = {"Bleu":bleu_result["bleu"],"Meteor":meteor_result["meteor"],"Rouge":rouge_result["rougeL"],"Accuracy_score":accuracy_score}
    result = {"Accuracy_score":accuracy_score}

    return result

