import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import torch
from flamingo_mini_task.utils import load_url
from flamingo_mini_task import FlamingoModel, FlamingoProcessor
from datasets import load_dataset,concatenate_datasets
from PIL import Image


flamingo_megatiny_captioning_models = {
    'flamingo-megatiny-opt': {
        'model': FlamingoModel.from_pretrained('landersanmi/flamingo-megatiny-opt'),
    },
    'flamingo-megatiny-opt-v2': {
        'model': FlamingoModel.from_pretrained('landersanmi/flamingo-megatiny-opt-v2'),
    }
}

def generate_text(image, model_name):
    model = flamingo_megatiny_captioning_models[model_name]['model']
    processor = FlamingoProcessor(model.config)

    prediction = model.generate_captions(images = image,
                                         processor = processor,
                                         prompt = "<image>",
                                        )

    return prediction[0]

image_input = gr.Image(value="coca-cola.jpg")
select_model = gr.inputs.Dropdown(choices=list(flamingo_megatiny_captioning_models.keys()))
text_output = gr.outputs.Textbox()

# Create the Gradio interface
gr.Interface(
    fn=generate_text,
    inputs=[image_input, 
            select_model
           ],
    outputs=text_output,
    title='Generate image captions [flamingo-megatiny`s]',
    description='Generate a text caption based on an image input with Flamingo megatiny models',
    theme='default'
).launch()