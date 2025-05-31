from flask import Flask, request, jsonify,render_template
from unsloth import FastLanguageModel
import torch
import re, json
import os

# === Load Unsloth model from Hugging Face ===
model_path = 'sue888888888888/essay_grader' # Replace with your model repo
max_seq_length = 2048
load_in_4bit = True
dtype = None  # Let it auto-select float16/32

# Load the model - if this fails, may need to specify the exact checkpoint
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# === Flask API ===
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("essay_grader_fixed.html")
@app.route("/grade", methods=["POST"])
def grade():
    data = request.get_json()
    question = data.get("question", "")
    reference = data.get("reference_answer", "")
    student = data.get("student_answer", "")
    mark_scheme = data.get("mark_scheme", {})

    prompt = f"""You are an expert essay grader. Grade the student's answer based on the mark scheme.

Question: {question}
Reference Answer: {reference}
Student Answer: {student}
Mark Scheme:
""" + "\n".join([f"{k}: {v}" for k, v in mark_scheme.items()]) + """

Return a JSON like this:
{
  "total_score": <score>,
  "max_score": <max>,
  "marks_awarded": [<mark_numbers>],
  "detailed_feedback": {
    "1": {"reason": "..."},
    ...
  }
}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("outputs:",outputs)
    print("decoded:",decoded)
    try:
        result_json = re.search(r'\{[\s\S]*\}', decoded).group(0)
        result = json.loads(result_json)
        return jsonify({"success": True, "result": result, "model_used": model_path})
    except Exception as e:
        return jsonify({"success": False, "error": f"Parsing error: {str(e)}", "raw_output": decoded})

if __name__ == "__main__":
    app.run(debug=True)
