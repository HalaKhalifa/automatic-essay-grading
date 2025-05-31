import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
import numpy as np

# Load fine-tuned model
model_name = "sue888888888888/essay_grader"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
FastLanguageModel.for_inference(model)

# Prompt template
prompt_template = """Below is an instruction that describes how to grade an essay, paired with an input that provides the grading schema. Write a response that grades essays based on the mark schema provided.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

def grade_essay(question, reference, student, mark1, mark2, mark3, mark4):
    mark_scheme = {
        "1": mark1,
        "2": mark2,
        "3": mark3,
        "4": mark4
    }

    instruction = "Grade this essay based on the following mark scheme:\n" + "\n".join([f"Criterion {k}: {v}" for k, v in mark_scheme.items()])
    input_text = f"Question: {question}\nReference Answer: {reference}\nStudent Answer: {student}"
    full_prompt = prompt_template.format(instruction=instruction, input_text=input_text)

    inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response = decoded.split("### Response:")[-1].strip()

    return response

# UI
demo = gr.Interface(
    fn=grade_essay,
    inputs=[
        gr.Textbox(label="Question"),
        gr.Textbox(label="Reference Answer"),
        gr.Textbox(label="Student Answer"),
        gr.Textbox(label="Marking Criterion 1"),
        gr.Textbox(label="Marking Criterion 2"),
        gr.Textbox(label="Marking Criterion 3"),
        gr.Textbox(label="Marking Criterion 4"),
    ],
    outputs=gr.Textbox(label="Model Response (Score or Explanation)"),
    title="📝 Essay Grader (Mistral + Unsloth)"
)

demo.launch()
