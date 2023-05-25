"""
utils
"""
import requests
import torch
from PIL import Image
from torch import nn
import torch.utils.data as data
from datasets import load_dataset,concatenate_datasets
from typing import List
from tqdm import tqdm
import numpy as np
import os
import json

def load_url(url: str):
    return Image.open(requests.get(url, stream=True).raw)


def load_image(path: str):
    return Image.open(path)


def unzip(l):
    return list(zip(*l))


class SquaredReLU(nn.Module):
    """ squared ReLU activation function"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.pow(torch.relu(x), 2)


def FeedForward(dim, mult=4, act='gelu'):
    """
    lucidrains implementation, slightly modified with the act parameter.
    """
    
    acts = dict(
        gelu=nn.GELU,
        sqrelu=SquaredReLU,
        relu=nn.ReLU
    )
    
    assert act in acts, f"act. can only be one of {acts.keys()}"
    
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        acts[act](),
        nn.Linear(inner_dim, dim, bias=False)
    )


def get_common_prefix_length(x: torch.Tensor) -> int:
    # assuming that x is a matrix
    try:
        return (x[0] == x[1:]).all(dim=0).tolist().index(False)
    except ValueError:
        return x.size(1)
    
class BilbaoCaptions(data.Dataset):
    #Clase parecida al de COCO pero adaptada a nuestro formato
    def __init__(self, dataset:List[str],split_name:str,transform=None,target_transform=None):
        lista_datasets=[]
        for idx,path in enumerate(dataset):
            lista_datasets.append(load_dataset(path,split=split_name))
            if idx >0:
                assert lista_datasets[idx-1].features.type == lista_datasets[idx].features.type
          
        self.dataset=concatenate_datasets(lista_datasets)
        self.transform = transform
        self.target_transform = target_transform
        self.new_size = (224, 224)
        self._resize_images(self.new_size)
        self.dataset=self.dataset.shuffle(seed=42)
        self.images= np.array([ image for image in tqdm(self.dataset["image"])],dtype=object)
    def _resize_images(self, new_size):
        for data_point in tqdm(self.dataset):
            image = data_point['image']            
            resized_image = image.resize(new_size)
            data_point['image'] = resized_image

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        
        target = self.dataset["caption"][index]

        
        # img = self.dataset["image"][index]
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    def __getcaption__(self,index):
        return self.dataset["caption"][index]
    def __len__(self):
        return len(self.dataset)


class VizWizCaptioningDataset(data.Dataset):
    def __init__(self, image_dir, annotation_dir, split, transform=None, target_transform=None):
        self.image_dir = image_dir + '/' + split
        self.annotation_file = os.path.join(annotation_dir, f"{split}.json")
        self.annotations = self.load_annotations()
        self.image_id_to_captions = self.build_caption_mapping()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, index):
        image_info = self.annotations['images'][index]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        
        target = self.__get_captions__(index)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
    
    def __get_captions__(self, image_id):
        item_info = self.annotations['images'][image_id]
        original_id = item_info['id']
        return self.image_id_to_captions[original_id]
    
    def build_caption_mapping(self):
        image_id_to_captions = {}
        annotations = self.annotations['annotations']
        for annotation in annotations:
            image_id = annotation['image_id']
            caption = annotation['caption']
            if image_id in image_id_to_captions:
                image_id_to_captions[image_id].append(caption)
            else:
                image_id_to_captions[image_id] = [caption]
        return image_id_to_captions
    
    def load_annotations(self):
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        return annotations


