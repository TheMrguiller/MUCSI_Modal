from typing import Optional, List, Dict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import CocoCaptions
# from pycocoevalcap.eval import COCOEvalCap
from flamingo_mini.utils import BilbaoCaptions
from flamingo_mini import FlamingoModel, FlamingoProcessor
import evaluate

class MyDatasetWrapper(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        image_id = index
        # print(image.shape)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(image_id)
        return image_id, image


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
@torch.no_grad()
def evaluate_image_captioning( #https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py
    dataset: BilbaoCaptions,
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
    captions_ = []
    ref_captions =[]
    wrapper = MyDatasetWrapper(dataset)
    
    wrapper = Subset(wrapper, range(start, end if end is not None else len(wrapper)))
    kwargs = {'num_workers': 1, 'pin_memory': True} 
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_loader = torch.utils.data.DataLoader(wrapper, batch_size=batch_size,shuffle=False,drop_last=False, **kwargs)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    # print(train_loader.device)
    loader = DataLoader(
        wrapper, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=num_workers, persistent_workers=True)
    
    # print(loader.dataset[0])
    for image_ids, pixels in tqdm(loader):
        print(len(image_ids),len(pixels))
        captions = model.generate_captions(
            processor, 
            pixel_values=pixels.to(model.device),
            prompt=prefix,
            device=device
        )
        # print(captions)
        # print(dataset.__getcaption__(image_ids))
        # captions.append(captions)
        # ref_captions.append(dataset.__getcaption__(image_ids))
        for image_id, caption in zip(image_ids.tolist(), captions):
            captions_.append(caption)
            ref_captions.append(dataset.__getcaption__(image_id))
        # if image_ids >0:
        #     break
            
    
    #Evaluate based in meteor,rouge.Novel metrics cider y spider
    bleu_metric = evaluate.load("bleu")
    bleu_result = bleu_metric.compute(predictions=captions_, references=ref_captions)
    meteor_metric = evaluate.load('meteor')
    meteor_result=meteor_metric.compute(predictions=captions_, references=ref_captions)
    rougue_metric=evaluate.load('rouge')
    rouge_result=rougue_metric.compute(predictions=captions_, references=ref_captions)
    # coco_result = dataset.coco.loadRes(results)
    # coco_eval = COCOEvalCap(dataset.coco, coco_result)
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # coco_eval.evaluate()
    result = {"Bleu":bleu_result["bleu"],"Meteor":meteor_result["meteor"],"Rouge":rouge_result["rougeL"]}
    print(result)
    return result

