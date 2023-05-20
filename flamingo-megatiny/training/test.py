import matplotlib.pyplot as plt
import torch

from flamingo_mini import FlamingoModel, FlamingoProcessor
from datasets import load_dataset
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = FlamingoModel.from_pretrained('flamingo-coco/checkpoint-22179')
model = FlamingoModel.from_pretrained('landersanmi/flamingo-megatiny')
model.to(device)
model.eval()
processor = FlamingoProcessor(model.config)
#model.push_to_hub("landersanmi/flamingo-megatiny")

dataset=load_dataset("landersanmi/BilbaoCaptions2",)

caption = model.generate_captions(processor, images=[dataset["train"]["image"][400]])
print('generated caption:')
print(caption)
print('ref caption:')
print(dataset["train"]["caption"][400])
plt.imshow(dataset["train"]["image"][400])
plt.show()
caption = model.generate_captions(processor, images=[dataset["test"]["image"][0]])
print('generated caption:')
print(caption)
print('ref caption:')
print(dataset["test"]["caption"][0])
plt.imshow(dataset["test"]["image"][0])
plt.show()
caption = model.generate_captions(processor, images=[dataset["test"]["image"][400]])
print('generated caption:')
print(caption)
print('ref caption:')
print(dataset["test"]["caption"][400])
plt.imshow(dataset["test"]["image"][400])
plt.show()
image = Image.open("Plaza_Mayor_de_Madrid_06.jpg")
caption = model.generate_captions(processor, prompt="<image>", images=[image])
print('generated caption:')
print(caption)
plt.imshow(image)
plt.show()
