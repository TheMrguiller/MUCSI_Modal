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
    'flamingo-tiny-scienceQA[COT+QA]': {
        'model': FlamingoModel.from_pretrained('TheMrguiller/Flamingo-tiny_ScienceQA_COT-QA'),
    },
    'flamingo-mini-bilbaocaptions-scienceQA[QA]': {
        'model': FlamingoModel.from_pretrained('TheMrguiller/Flamingo-mini-Bilbao_Captions-task_BilbaoQA-ScienceQA'),
    },
    'flamingo-megatiny-opt-scienceQA[QA]':{
        'model': FlamingoModel.from_pretrained('landersanmi/flamingo-megatiny-opt-QA')
    },
}


def generate_text(image, question, option_a, option_b, option_c, option_d, cot_checkbox, model_name):
    model = flamingo_megatiny_captioning_models[model_name]['model']
    processor = FlamingoProcessor(model.config)

    prompt = ""
    if cot_checkbox:
        prompt += "[COT]"
    else:
        prompt += "[QA]"
    
    prompt += "[CONTEXT]<image>[QUESTION]{} [OPTIONS] (A) {} (B) {} (C) {} (D) {} [ANSWER]".format(question,
                                                                                                   option_a,
                                                                                                   option_b,
                                                                                                   option_c,
                                                                                                   option_d)

    print(prompt)
    prediction = model.generate_captions(images = image,
                                         processor = processor,
                                         prompt = prompt,
                                        )

    return prediction[0]




image_input = gr.Image(value="giraffes.jpg")
question_input = gr.inputs.Textbox(default="Which animal is this?")
opt_a_input = gr.inputs.Textbox(default="Dog")
opt_b_input = gr.inputs.Textbox(default="Cat")
opt_c_input = gr.inputs.Textbox(default="Giraffe")
opt_d_input = gr.inputs.Textbox(default="Horse")
cot_checkbox = gr.inputs.Checkbox(label="Generate COT")
select_model = gr.inputs.Dropdown(choices=list(flamingo_megatiny_captioning_models.keys()))

text_output = gr.outputs.Textbox()

# Create the Gradio interface
gr.Interface(
    fn=generate_text,
    inputs=[image_input, 
            question_input, 
            opt_a_input, 
            opt_b_input, 
            opt_c_input, 
            opt_d_input, 
            cot_checkbox,
            select_model
           ],
    outputs=text_output,
    title='Generate answers from MCQ',
    description='Generate answers from Multiple Choice Questions or generate a Chain Of Though about the question and the options given',
    theme='default'
).launch()