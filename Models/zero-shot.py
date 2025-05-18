# zero-shot.py

import json
import requests
from tqdm import tqdm

OLLAMA_MODEL = "mistral:7b-instruct-v0.2-q2_K"
OLLAMA_URL = "http://localhost:11434/api/generate"
DATA_PATH = "data/essay-grading-criteria.json"
OUTPUT_PATH = "results/zero_shot_results.json"

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

def call_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"]

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

def main():
    with open(DATA_PATH) as f:
        dataset = json.load(f)

    results = []

    for item in tqdm(dataset):
        prompt = build_prompt(item)
        response_text = call_ollama(prompt)
        predicted_score, explanation = parse_score(response_text)

        results.append({
            "question": item["question"],
            "student_answer": item["student_answer"],
            "true_score": item["score"],
            "predicted_score": predicted_score,
            "rationale": explanation,
            "raw_response": response_text
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()