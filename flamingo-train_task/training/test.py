import matplotlib.pyplot as plt
import torch
from flamingo_mini_task.utils import load_url
from flamingo_mini_task import FlamingoModel, FlamingoProcessor
from datasets import load_dataset,concatenate_datasets
from train import prepare_evaluation_dataset_BilbaoQA
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = FlamingoModel.from_pretrained('/home/d4k/Documents/guillermo/MUCSI_Modal/flamingo-train_task/training/flamingo-Bilbao/checkpoint-9860')

# model.to(device)
# model.eval()
# processor = FlamingoProcessor(model.config)
model.push_to_hub("TheMrguiller/Flamingo-tiny_ScienceQA_COT")
# # dataset=load_dataset("landersanmi/BilbaoCaptions2",)
# dataset=load_dataset("TheMrguiller/ScienceQA")
# dataset=prepare_evaluation_dataset_BilbaoQA(model.config,["TheMrguiller/ScienceQA"],"test")
# img, target, label=dataset[95]
# target = target.replace("COT","QA")
# img=dataset.dataset["image"][95]
# print('ref question:')
# print(target)
# print("label")
# print(label)
# print('ref caption:')
# print(label)
# plt.imshow(img)
# plt.show()
# caption = model.generate_captions(processor, images=img,prompt=target)
# print('generated caption:')
# print(caption)
# target = target.replace("QA","COT")
# caption = model.generate_captions(processor, images=img,prompt=target)
# print('generated caption:')
# print(caption)


# caption = model.generate_captions(processor, images=img,prompt="<image>")
# print('generated caption:')
# print(caption)
# caption = model.generate_captions(processor, images=[dataset["test"]["image"][0]])
# print('generated caption:')
# print(caption)
# print('ref caption:')
# print(dataset["test"]["caption"][0])
# plt.imshow(dataset["test"]["image"][0])
# plt.show()
# caption = model.generate_captions(processor, images=[dataset["test"]["image"][400]])
# print('generated caption:')
# print(caption)
# print('ref caption:')
# print(dataset["test"]["caption"][400])
# plt.imshow(dataset["test"]["image"][400])
# plt.show()
# caption = model.generate_captions(processor, images=[dataset["test"]["image"][150]])
# print('generated caption:')
# print(caption)
# print('ref caption:')
# print(dataset["test"]["caption"][150])
# plt.imshow(dataset["test"]["image"][150])
# plt.show()
# caption = model.generate_captions(processor, images=[dataset["test"]["image"][200]])
# print('generated caption:')
# print(caption)
# print('ref caption:')
# print(dataset["test"]["caption"][200])
# plt.imshow(dataset["test"]["image"][200])
# plt.show()
# image = load_url('https://interrailero.com/wp-content/uploads/2022/01/que-ver-en-barcelona-mapa.jpg')
# caption = model.generate_captions(processor, images=[image])
# print('generated caption:')
# print(caption)
# plt.imshow(image)
# plt.show()