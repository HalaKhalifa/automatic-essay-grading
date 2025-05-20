import json
from mlx_lm import load, generate
from tqdm import tqdm

# Load Mistral-7B Instruct 4-bit with mlx
model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.2-4bit")

# Define paths
DATA_PATH = "data/essay-grading-criteria.json"
OUTPUT_PATH = "results/zero_shot_results_mlx.json"

# Prompt template
def build_prompt(item):
    mark_scheme_text = "\n".join([f"{k}. {v}" for k, v in item["mark_scheme"].items()])
    prompt = (
        f"Question: {item['question']}\n"
        f"Reference Answer: {item['reference_answer']}\n"
        f"Mark Scheme:\n{mark_scheme_text}\n"
        f"Student Answer: {item['student_answer']}\n"
        f"Instruction: {item['instruction']}\n\n"
        f"Return only the predicted score (1–4) and a short explanation in this format:\n"
        f"Score: <number>\nExplanation: <reason>\n"
    )
    return prompt

# Parse model output to extract score and explanation
def parse_score(response_text):
    try:
        lines = response_text.strip().splitlines()
        score_line = next(line for line in lines if "Score:" in line)
        score = int(score_line.strip().split(":")[-1])
        explanation_line = next(line for line in lines if "Explanation:" in line)
        explanation = explanation_line.strip().split(":", 1)[-1].strip()
        return score, explanation
    except Exception as e:
        print("Error parsing response:", e)
        return -1, "Failed to parse"

# Load JSON dataset
with open(DATA_PATH, "r") as f:
    dataset = json.load(f)

# Run inference
results = []
for item in tqdm(dataset):
    prompt = build_prompt(item)

    # Generate response
    response = generate(model, tokenizer, prompt, max_tokens=200)
    response_text = response.strip()

    # Parse score and rationale
    predicted_score, explanation = parse_score(response_text)

    # Store results
    results.append({
        "question": item["question"],
        "student_answer": item["student_answer"],
        "true_score": item["score"],
        "predicted_score": predicted_score,
        "rationale": explanation,
        "raw_response": response_text
    })

# Save to JSON
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n Results saved to {OUTPUT_PATH}")