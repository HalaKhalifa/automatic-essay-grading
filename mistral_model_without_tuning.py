# Commented out IPython magic to ensure Python compatibility.
# %%capture
# # Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes

"""* We support Llama, Mistral, CodeLlama, TinyLlama, Vicuna, Open Hermes etc
* And Yi, Qwen ([llamafied](https://huggingface.co/models?sort=trending&search=qwen+llama)), Deepseek, all Llama, Mistral derived archs.
* We support 16bit LoRA or 4bit QLoRA. Both 2x faster.
* `max_seq_length` can be set to anything, since we do automatic RoPE Scaling via [kaiokendev's](https://kaiokendev.github.io/til) method.
* With [PR 26037](https://github.com/huggingface/transformers/pull/26037), we support downloading 4bit models **4x faster**! [Our repo](https://huggingface.co/unsloth) has Llama, Mistral 4bit models.
* [**NEW**] We make Gemma 6 trillion tokens **2.5x faster**! See our [Gemma notebook](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing)
"""


from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

# from transformers import BitsAndBytesConfig

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit", # New Google 6 trillion tokens model 2.5x faster!
    "unsloth/gemma-2b-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,

)


prompt = """Below is an instruction that describes how to grade an essay, paired with an input that provides the grading schema. Write a response that grades essays based on the mark schema provided.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Add EOS token to stop generation

def formatting_prompts_func(examples):
    texts = []
    for q, ra, sa, ms, score in zip(examples["question"], examples["reference_answer"], examples["student_answer"], examples["mark_scheme"], examples["score"]):
        # Convert mark_scheme dict to string
        mark_scheme_str = "\n".join([f"{k}: {v}" for k, v in ms.items()])
        instruction = "Grade this essay based on the following mark scheme:\n" + mark_scheme_str
        input_text = f"Question: {q}\nReference Answer: {ra}\nStudent Answer: {sa}"
        output_text = str(score)

        # Format full prompt
        text = prompt.format(instruction, input_text, output_text) + EOS_TOKEN
        texts.append(text)
    return { "text": texts }

# Load dataset and apply formatting
dataset = load_dataset("sue888888888888/essay-grading-criteria", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)



def zero_shot_evaluate(model, tokenizer, dataset, num_samples=20):
    model.eval()
    predictions = []
    targets = []

    for example in dataset.select(range(num_samples)):
        q = example["question"]
        ra = example["reference_answer"]
        sa = example["student_answer"]
        ms = example["mark_scheme"]
        true_score = example["score"]

        # Build prompt
        mark_scheme_str = "\n".join([f"{k}: {v}" for k, v in ms.items()])
        instruction = "Grade this essay based on the following mark scheme:\n" + mark_scheme_str
        input_text = f"Question: {q}\nReference Answer: {ra}\nStudent Answer: {sa}"
        full_prompt = prompt.format(instruction, input_text, "") + EOS_TOKEN

        # Tokenize and run
        inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=32)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract only the predicted score (e.g., number like 3 or 4)
        import re
        match = re.search(r"\d+", decoded)
        if match:
            pred_score = int(match.group())
            predictions.append(pred_score)
            targets.append(true_score)

    return predictions, targets


def evaluate_scores(predictions, targets):
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    acc = accuracy_score(targets, predictions)

    print("📊 Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Exact Match Accuracy: {acc*100:.2f}%")

def main():
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(
    [
        prompt.format(
            "Grade this essay based on the following mark scheme:\n1: Defines photosynthesis correctly\n2: Mentions sunlight as an energy source\n3: Includes carbon dioxide and water as inputs\n4: Mentions oxygen or glucose as products ", # instruction

            "What is photosynthesis?\nReference Answer: Photosynthesis is the process by which green plants make their own food using sunlight, carbon dioxide, and water. The process occurs in the chloroplasts and produces glucose and oxygen as end products.\nStudent Answer: Photosynthesis is when plants eat sunlight and turn it into food and air.", # input

            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    tokenizer.batch_decode(outputs)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("🧠 Model Prediction:\n", decoded_output[0])

    print("\n")
    predictions, targets = zero_shot_evaluate(model, tokenizer, dataset, num_samples=50)
    evaluate_scores(predictions, targets)


if __name__ == "__main__":
    main()

##@title Show current memory stats
    # gpu_stats = torch.cuda.get_device_properties(0)
    # start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    # max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    # print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    # print(f"{start_gpu_memory} GB of memory reserved.")

    # 
    # #@title Show final memory and time stats
    # used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    # used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    # used_percentage = round(used_memory         /max_memory*100, 3)
    # lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    # print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    # print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    # print(f"Peak reserved memory = {used_memory} GB.")
    # print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    # print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    # print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")