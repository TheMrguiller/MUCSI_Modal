import matplotlib.pyplot as plt
import torch
import random

from flamingo_mini import FlamingoModel, FlamingoProcessor
from datasets import load_dataset
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlamingoModel.from_pretrained('flamingo-coco-opt-domain/checkpoint-5808')
#model = FlamingoModel.from_pretrained('landersanmi/flamingo-megatiny-opt')
model.to(device)
model.eval()
processor = FlamingoProcessor(model.config)
#model.push_to_hub("landersanmi/flamingo-megatiny-opt")

dataset=load_dataset("landersanmi/BilbaoCaptions2",)

for i in range(10):
    print(i)
    rand_index = random.randint(0, len(dataset['test']['image'])-1)
    caption = model.generate_captions(processor, images=[dataset["test"]["image"][rand_index]])
    print('generated caption:')
    print(caption)
    print('ref caption:')
    print(dataset["test"]["caption"][rand_index])
    plt.imshow(dataset["test"]["image"][rand_index])
    plt.show()

caption = model.generate_captions(processor, images=[dataset["test"]["image"][399]])
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



'''
image = Image.open("Plaza_Mayor_de_Madrid_06.jpg")
caption = model.generate_captions(processor, prompt="<image>", images=[image])
print('generated caption:')
print(caption)
plt.imshow(image)
plt.show()
'''