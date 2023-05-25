from typing import Optional, List, Dict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import CocoCaptions
from pycocoevalcap.eval import COCOEvalCap

from flamingo_mini import FlamingoModel, FlamingoProcessor
from flamingo_mini.utils import BilbaoCaptions, VizWizCaptioningDataset
import evaluate
#from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
#from pycocoevalcap.cider.cider import Cider
#from pycocoevalcap.spice.spice import Spice


class MyCocoDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        image_id = self.dataset.ids[index]
        return image_id, image

class MyVizWIzDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        image_id = index
        return image_id, image

class MyBilbaoDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        image_id = index
        return image_id, image


@torch.no_grad()
def evaluate_image_captioning_coco( #https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py
    dataset: CocoCaptions,
    model: FlamingoModel, 
    *,
    prefix: str = "<image>",
    start = 0,
    end: Optional[int] = None,
    verbose: bool = True,
    batch_size: int = 64,
    num_workers: int = 8, 
) -> Dict[str, float]:

    processor = FlamingoProcessor(model.config)
    results: List[dict] = []

    wrapper = MyCocoDatasetWrapper(dataset)
    wrapper = Subset(wrapper, range(start, end if end is not None else len(wrapper)))
    loader = DataLoader(
        wrapper, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=num_workers)

    for image_ids, pixels in tqdm(loader, disable=not verbose):
        captions = model.generate_captions(
            processor, 
            pixel_values=pixels.to(model.device),
            prompt=prefix
        )
        
        for image_id, caption in zip(image_ids.tolist(), captions):
            results.append(dict(image_id=image_id, caption=caption))

    coco_result = dataset.coco.loadRes(results)
    coco_eval = COCOEvalCap(dataset.coco, coco_result)
    print(dataset.coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()
    return coco_eval.eval

@torch.no_grad()
def evaluate_image_captioning_vizwiz(
    dataset: VizWizCaptioningDataset,
    model: FlamingoModel, 
    *,
    prefix: str = "<image>",
    start = 0,
    end: Optional[int] = None,
    verbose: bool = True,
    batch_size: int = 64,
    num_workers: int = 8, 
) -> Dict[str, float]:

    processor = FlamingoProcessor(model.config)
    captions_ = []
    ref_captions =[]

    wrapper = MyVizWIzDatasetWrapper(dataset)
    wrapper = Subset(wrapper, range(start, end if end is not None else len(wrapper)))
    loader = DataLoader(
        wrapper, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=num_workers, persistent_workers=True)

    for image_ids, pixels in tqdm(loader, disable=not verbose):
        generated_captions = model.generate_captions(
            processor, 
            pixel_values=pixels.to(model.device),
            prompt=prefix
        )

        for image_id, caption in zip(image_ids.tolist(), generated_captions):
            captions_.append(caption)
            ref_captions.append(dataset.__get_captions__(image_id))


    #Evaluate based in bleu, meteor, rouge.
    bleu_metric = evaluate.load("bleu")
    bleu_result = bleu_metric.compute(predictions=captions_, references=ref_captions)
    meteor_metric = evaluate.load('meteor')
    meteor_result=meteor_metric.compute(predictions=captions_, references=ref_captions)
    rougue_metric=evaluate.load('rouge')
    rouge_result=rougue_metric.compute(predictions=captions_, references=ref_captions)

    bleu = bleu_result['bleu']
    bleu1 = bleu_result['precisions'][0]
    bleu2 = bleu_result['precisions'][1]
    bleu3 = bleu_result['precisions'][2]
    bleu4 = bleu_result['precisions'][3]
    rouge_l = rouge_result['rougeL']
    meteor = meteor_result['meteor']

    result = {"Bleu":bleu,
              "Bleu_1":bleu1, 
              "Bleu_2":bleu2, 
              "Bleu_3":bleu3, 
              "Bleu_4":bleu4, 
              "ROUGE_L":rouge_l,
              "METEOR":meteor
              }
    
    return result

@torch.no_grad()
def evaluate_image_captioning_Bilbao(
    dataset: BilbaoCaptions,
    model: FlamingoModel, 
    *,
    prefix: str = "<image>",
    start = 0,
    end: Optional[int] = None,
    verbose: bool = True,
    batch_size: int = 64,
    num_workers: int = 8, 
) -> Dict[str, float]:

    processor = FlamingoProcessor(model.config)
    captions_ = []
    ref_captions =[]
    wrapper = MyBilbaoDatasetWrapper(dataset)
    
    wrapper = Subset(wrapper, range(start, end if end is not None else len(wrapper)))
    kwargs = {'num_workers': 1, 'pin_memory': True} 
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_loader = torch.utils.data.DataLoader(wrapper, batch_size=batch_size,shuffle=False,drop_last=False, **kwargs)
    
    # print(train_loader.device)
    loader = DataLoader(
        wrapper, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=num_workers, persistent_workers=True)
    #tokenizer = PTBTokenizer()
    #gts = {} #dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
    #res = {} #dictionary with key <image> and value <tokenized reference sentence>
    
    # print(loader.dataset[0])
    for image_ids, pixels in tqdm(loader):
        print(len(image_ids),len(pixels))
        #images = 
        captions = model.generate_captions(
            processor, 
            pixel_values=pixels.to(model.device),
            prompt=prefix,
            #device=device
        )
        # print(captions)
        # print(dataset.__getcaption__(image_ids))
        # captions.append(captions)
        # ref_captions.append(dataset.__getcaption__(image_ids))
        for image_id, caption in zip(image_ids.tolist(), captions):
            captions_.append(caption)
            ref_captions.append(dataset.__getcaption__(image_id))

            #ref_caption=dataset.__getcaption__(image_id)
            #ref_captions.append(ref_caption)
            # print(caption)
            # print(image_id)
            #gts[image_id]= {"caption":caption}
            #res[image_id]= {"caption":ref_caption}
        # if image_ids >0:
        #   break
            
    #gts  = tokenizer.tokenize(gts)
    #res = tokenizer.tokenize(res)
    #cider_metric=Cider()
    #spice_metric=Spice()
    #cider_result,_=cider_metric.compute_score(gts=gts,res=res)
    #print(f"CIDeR:{cider_result}")

    #Evaluate based in meteor,rouge.Novel metrics cider y spider
    bleu_metric = evaluate.load("bleu")
    bleu_result = bleu_metric.compute(predictions=captions_, references=ref_captions)
    meteor_metric = evaluate.load('meteor')
    meteor_result=meteor_metric.compute(predictions=captions_, references=ref_captions)
    rougue_metric=evaluate.load('rouge')
    rouge_result=rougue_metric.compute(predictions=captions_, references=ref_captions)

    result = {"Bleu":bleu_result["bleu"],"Meteor":meteor_result["meteor"],"Rouge":rouge_result["rougeL"]}
    #result = {"Bleu":bleu_result["bleu"],"Meteor":meteor_result["meteor"],"Rouge":rouge_result["rougeL"],"CIDEr":cider_result}

    return result