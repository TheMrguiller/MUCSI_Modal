import matplotlib.pyplot as plt
import torch
import random
import os

from flamingo_mini import FlamingoModel, FlamingoProcessor
from datasets import load_dataset
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model1 = FlamingoModel.from_pretrained('landersanmi/flamingo-megatiny-opt')
#model = FlamingoModel.from_pretrained('TheMrguiller/Flamingo-tiny-Bilbao_Captions')
#model = FlamingoModel.from_pretrained('TheMrguiller/Flamingo-mini-Bilbao_Captions')
model1 = FlamingoModel.from_pretrained('landersanmi/flamingo-megatiny-opt-v2')
#model1 = FlamingoModel.from_pretrained('dhansmair/flamingo-mini')
model1.to(device)
model1.eval()
processor1 = FlamingoProcessor(model1.config)

model2 = FlamingoModel.from_pretrained('flamingo-coco-opt-domain/checkpoint-7744')
model2.to(device)
model2.eval()
processor2 = FlamingoProcessor(model2.config)
model2.push_to_hub("landersanmi/flamingo-megatiny-opt-domain")

model3 = FlamingoModel.from_pretrained('flamingo-coco-opt-vizwiz-domain/checkpoint-7744')
model3.to(device)
model3.eval()
processor3 = FlamingoProcessor(model3.config)
model3.push_to_hub("landersanmi/flamingo-megatiny-opt-v2-domain")

IMAGE_IDS = [56, 107, 405, 504, 576, 607, 630]


dataset1=load_dataset("landersanmi/BilbaoCaptions2",)
dataset2=load_dataset("TheMrguiller/BilbaoCaptions",)
datasets = [dataset1, dataset2]

for dataset in datasets:
    for idx in IMAGE_IDS:
        print(idx)
        caption1 = model1.generate_captions(processor1, images=[dataset["test"]["image"][idx]])
        caption2 = model2.generate_captions(processor2, images=[dataset["test"]["image"][idx]])
        caption3 = model3.generate_captions(processor3, images=[dataset["test"]["image"][idx]])
        print('generated caption flamingo-megatiny-opt-v2')
        print(caption1)
        print('generated caption coco checkpoint-7744')
        print(caption2)
        print('generated caption coco-vizwiz checkpoint-7744')
        print(caption3)   
        print('ref caption:')
        print(dataset["test"]["caption"][idx])
        plt.imshow(dataset["test"]["image"][idx])
        plt.show()


folder_path = "imgs_test"  # Replace with the path to your folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    image = Image.open(file_path)
    caption1 = model1.generate_captions(processor1, prompt="<image>", images=[image])
    caption2 = model2.generate_captions(processor2, prompt="<image>", images=[image])
    caption3 = model3.generate_captions(processor3, prompt="<image>", images=[image])
    print('generated caption flamingo-megatiny-opt-v2')
    print(caption1)
    print('generated caption coco checkpoint-7744')
    print(caption2)
    print('generated caption coco-vizwiz checkpoint-7744')
    print(caption3)   
    plt.imshow(image)
    plt.show()
