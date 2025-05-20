# zero_shot_evaluation_metrics.py
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, confusion_matrix, classification_report, cohen_kappa_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def rationale_similarity(true_rats, pred_rats):
    # Filter empty rationales
    paired = [(t, p) for t, p in zip(true_rats, pred_rats) if t.strip() and p.strip()]
    if not paired:
        return None
    
    true_texts, pred_texts = zip(*paired)
    vectorizer = TfidfVectorizer(stop_words='english')
    true_vecs = vectorizer.fit_transform(true_texts)
    pred_vecs = vectorizer.transform(pred_texts)
    
    similarities = []
    for i in range(true_vecs.shape[0]):
        sim = cosine_similarity(true_vecs[i], pred_vecs[i])[0][0]
        similarities.append(sim)
    return np.mean(similarities)

def load_baseline_scores(path):
    baseline_data = load_jsonl(path)
    baseline_data = [item for item in baseline_data if isinstance(item.get("predicted_score"), int)]
    return [int(item["predicted_score"]) for item in baseline_data]

def compute_metrics(true, pred):
    return {
        "accuracy": accuracy_score(true, pred),
        "mae": mean_absolute_error(true, pred),
        "qwk": cohen_kappa_score(true, pred, weights='quadratic'),
        "classification_report": classification_report(true, pred, zero_division=0, output_dict=True)
    }

def main():
    # Load main results
    results_path = "results/zero_shot_results_mlx_2.json"
    data = load_jsonl(results_path)
    
    # Filter valid entries with numeric predicted scores
    data = [item for item in data if isinstance(item.get("predicted_score"), int) and item["predicted_score"] >= 0]

    true_scores = [int(item["true_score"]) for item in data]
    predicted_scores = [int(item["predicted_score"]) for item in data]

    true_rationales = [item.get("true_rationale", "") for item in data]
    predicted_rationales = [item.get("predicted_rationale", "") for item in data]

    # Compute main metrics
    accuracy = np.mean(np.array(true_scores) == np.array(predicted_scores))
    mae = mean_absolute_error(true_scores, predicted_scores)
    qwk = cohen_kappa_score(true_scores, predicted_scores, weights='quadratic')
    class_report = classification_report(true_scores, predicted_scores, zero_division=0)

    rationale_sim = rationale_similarity(true_rationales, predicted_rationales)

    # Load baselines 'for Phase 2'
    baseline_gpt35_path = "gpt35-baseline-predictions.jsonl"
    baseline_bert_path = "bert-baseline-predictions.jsonl"

    try:
        gpt35_scores = load_baseline_scores(baseline_gpt35_path)
        bert_scores = load_baseline_scores(baseline_bert_path)
        gpt35_metrics = compute_metrics(true_scores, gpt35_scores)
        bert_metrics = compute_metrics(true_scores, bert_scores)
    except FileNotFoundError:
        gpt35_metrics = None
        bert_metrics = None

    # Confusion matrix plot
    cm = confusion_matrix(true_scores, predicted_scores)
    labels = sorted(list(set(true_scores + predicted_scores)))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Score")
    plt.ylabel("True Score")
    plt.title("Confusion Matrix")
    plt.show()

    # Error distribution histogram
    errors = np.array(true_scores) - np.array(predicted_scores)
    plt.hist(errors, bins=range(min(errors), max(errors)+2), align='left', color='skyblue')
    plt.xlabel("Error (True - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.show()

    # Print mismatches (first 3 examples)
    mismatches = [item for item in data if int(item["true_score"]) != int(item["predicted_score"])]
    print(f"Total mismatches: {len(mismatches)} out of {len(data)} samples.")
    for i, sample in enumerate(mismatches[:3], 1):
        print(f"\nMismatch #{i}:")
        print(f"True score: {sample['true_score']} | Predicted score: {sample['predicted_score']}")
        print(f"True rationale: {sample.get('true_rationale','N/A')}")
        print(f"Predicted rationale: {sample.get('predicted_rationale','N/A')}")

    # Final summary
    print("\n=== Final Evaluation Summary ===")
    print(f"Exact Match Accuracy: {accuracy:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")
    print(f"Detailed Classification Report:\n{class_report}")
    if rationale_sim is not None:
        print(f"Average Rationale Cosine Similarity: {rationale_sim:.4f}")
    else:
        print("No rationales available to compute similarity.")

    if gpt35_metrics:
        print("\nGPT-3.5 Baseline Metrics:")
        print(f"  Accuracy: {gpt35_metrics['accuracy']:.4f}")
        print(f"  MAE: {gpt35_metrics['mae']:.4f}")
        print(f"  QWK: {gpt35_metrics['qwk']:.4f}")

    if bert_metrics:
        print("\nBERT Baseline Metrics:")
        print(f"  Accuracy: {bert_metrics['accuracy']:.4f}")
        print(f"  MAE: {bert_metrics['mae']:.4f}")
        print(f"  QWK: {bert_metrics['qwk']:.4f}")

if __name__ == "__main__":
    main()
