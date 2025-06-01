from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import re, json
import os
from dotenv import load_dotenv
load_dotenv()

# === Hugging Face Model Settings ===
model_path = "sue888888888888/essay_grader" 
HF_API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
print("Using Hugging Face token:", HF_API_TOKEN)

# === Flask API ===
app = Flask(__name__)
CORS(app)

# === Helper Function: Call HF Inference API ===
def query_huggingface_model(prompt, model_id=model_path):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model_id}",
        headers=headers,
        json={"inputs": prompt}
    )
    return response.json()

@app.route("/", methods=["GET"])
def index():
    return render_template("essay_grader_fixed.html")

@app.route('/', methods=['POST', 'OPTIONS'])
def grade():
    data = request.get_json()
    question = data.get("question", "")
    reference = data.get("reference_answer", "")
    student = data.get("student_answer", "")
    mark_scheme = data.get("mark_scheme", {})

    prompt = f"""
You are an expert essay grader. Grade the student's answer using the mark scheme below.

Respond ONLY with valid JSON in this format:
{{
  "total_score": <int>,
  "max_score": <int>,
  "marks_awarded": [<int>, ...],
  "detailed_feedback": {{
    "1": {{"reason": "<string>"}},
    "2": {{"reason": "<string>"}},
    ...
  }}
}}

Example:

Question: {question}
Reference Answer: {reference}
Student Answer: {student}

Mark Scheme:
""" + "\n".join([f"{k}: {v}" for k, v in mark_scheme.items()])


    # Call HF hosted model
    response = query_huggingface_model(prompt)
    print("=== HF raw output ===")
    print(response)
    try:
        decoded = response[0]["generated_text"]
        match = re.search(r'\{[\s\S]*\}', decoded)
        if not match:
            raise ValueError("JSON output not found in model response")
        result_json = match.group(0)
        result = json.loads(result_json)
        return jsonify({"success": True, "result": result, "model_used": model_path})
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Parsing error: {str(e)}",
            "raw_output": response
        })

if __name__ == "__main__":
    app.run(debug=True)
