from flask import Flask, request, jsonify, render_template
from unsloth import FastLanguageModel
import torch
import re, json
import os

from flask_cors import CORS

# === Flask API ===
app = Flask(__name__)
CORS(app)

# === Load Unsloth model from Hugging Face ===
model_path = "sue888888888888/essay_grader"  # Replace with your model repo
max_seq_length = 2048
load_in_4bit = True
dtype = None  # Let it auto-select float16/32

# Load the model - if this fails, may need to specify the exact checkpoint
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)




@app.route("/", methods=["GET"])
def index():
    return render_template("essay_grader_fixed.html")


@app.route("/grade", methods=["POST"])
def grade():

    print("in post")
    data = request.get_json()
    question = data.get("question", "")
    reference = data.get("reference_answer", "")
    student = data.get("student_answer", "")
    mark_scheme = data.get("mark_scheme", {})
    results=[]

    print(data)
    prompt = """Below is an instruction that describes how to grade an essay, paired with an input that provides the grading schema. Write a response that grades essays based on the mark schema provided.

### Instruction:
{}

### Input:
{}

### Response:
"""

    mark_scheme_str = "\n".join([f"{k}: {v}" for k, v in mark_scheme.items()])
    instruction = (
        "Grade this essay based on the following mark scheme:\n" + mark_scheme_str
    )
    input_text = f"Question: {question}\nReference Answer: {reference}\nStudent Answer: {student}"

    # Generate score
    test_prompt = prompt.format(instruction, input_text)

    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.1,  # Low temperature for more deterministic output
            top_p=0.9,
            do_sample=False,  # Greedy decoding for evaluation
            use_cache=True,
        )

    # Decode the output
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the generated score - look for the first number in the response
    response_part = decoded.split("### Response:")[-1].strip()

    # Extract numeric score using regex
    score_match = re.search(r'\b(\d+)\b', response_part)
    pred_score = int(score_match.group(1)) if score_match else None

    results.append({
        "question": question,
        "student_answer": student[:100] + "...",  # Truncate for display
        "pred_score": pred_score,
        "full_response": response_part
    })
    print("results:", results)
    # print("decoded:", decoded)
    try:
        result_json = re.search(r"\{[\s\S]*\}", decoded).group(0)
        result = json.loads(result_json)
        return jsonify({"success": True, "pred_score": result.get("total_score", pred_score)})
    except Exception as e:
        return jsonify({
            "success": True if pred_score is not None else False,
            "pred_score": pred_score,
            "error": None if pred_score is not None else f"Parsing error: {str(e)}",
        })



if __name__ == "__main__":
    app.run(debug=True)
