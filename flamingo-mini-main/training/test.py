import matplotlib.pyplot as plt
import torch

from flamingo_mini import FlamingoModel, FlamingoProcessor
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlamingoModel.from_pretrained('/home/d4k/Documents/guillermo/MUCSI_Modal/flamingo-mini-main/training/flamingo-Bilbao/checkpoint-1452')
model.to(device)
model.eval()
processor = FlamingoProcessor(model.config)
model.push_to_hub("TheMrguiller/Flamingo-tiny-Bilbao_Captions")
# dataset=load_dataset("landersanmi/BilbaoCaptions2",)
# caption = model.generate_captions(processor, images=[dataset["train"]["image"][400]])
# print('generated caption:')
# print(caption)
# print('ref caption:')
# print(dataset["train"]["caption"][400])
# plt.imshow(dataset["train"]["image"][400])
# plt.show()
# # caption = model.generate_captions(processor, images=[dataset["test"]["image"][0]])
# # print('generated caption:')
# # print(caption)
# # print('ref caption:')
# # print(dataset["test"]["caption"][0])
# # plt.imshow(dataset["test"]["image"][0])
# # plt.show()
# caption = model.generate_captions(processor, images=[dataset["test"]["image"][400]])
# print('generated caption:')
# print(caption)
# print('ref caption:')
# print(dataset["test"]["caption"][400])
# plt.imshow(dataset["test"]["image"][400])
# plt.show()