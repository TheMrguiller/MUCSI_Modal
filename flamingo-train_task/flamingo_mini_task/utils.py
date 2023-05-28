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
    
class BilbaoQA(data.Dataset):
    #Clase parecida al de COCO pero adaptada a nuestro formato
    def __init__(self, dataset:List[str],split_name:str,transform=None,target_transform=None):
        lista_datasets=[]
        for idx,path in enumerate(dataset):
            lista_datasets.append(load_dataset(path,split=split_name))
            #if idx >0:
                #assert lista_datasets[idx-1].features.type == lista_datasets[idx].features.type
          
        self.dataset=concatenate_datasets(lista_datasets)
        ##Eliminate those lines which doesnt have data in it
        print(f"Before preprocessing:{self.dataset}")
        self.dataset = self.dataset.filter(lambda value: value["question"]!="")
        self.dataset = self.dataset.filter(lambda value: value["image"]!=None)
        print(f"After preprocessing:{self.dataset}")
        self.transform = transform
        self.target_transform = target_transform
        self.new_size = (224, 224)
        self._resize_images(self.new_size)
        ## Duplicate those entries that have chainofthought in it in order to train in both task, and put the answer in blank
        self.datasetcot= self.dataset.filter(lambda value: value["CTH"]==False)
        #self.datasetcot=self.datasetcot.map(self.blank_proccess) #Ponemos en blanco para utilizarlo como flag
        self.dataset = concatenate_datasets([self.dataset,self.datasetcot])
        print(f"Total QA and COT:{self.dataset}")
        self.dataset=self.dataset.shuffle(seed=42)
        self.images= np.array([ image for image in tqdm(self.dataset["image"])],dtype=object)
    
    def blank_proccess(self,example):
        example["answer"] = ""
        return example
        
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
        
        
        target = self.dataset[index]
        if self.dataset["solution"][index]!="":
            label = self.dataset["solution"][index].replace("\n","")
        else:
            label = self.dataset["answer"][index]
        # img = self.dataset["image"][index]
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        #print(label)
        label = target + label
        return img, target, label
    
    def __getcaption__(self,index):
        return self.dataset["caption"][index]
    def __getanswer__(self,index):
        return self.dataset["answer"][index]
    def __len__(self):
        return len(self.dataset)
