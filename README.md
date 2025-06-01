# 📝 Automatic Essay Grading using Mistral-7B

An NLP project to automate essay grading using a fine-tuned transformer model on Hugging Face.

---

## 📚 Overview

Manual grading is time-consuming and inconsistent. This project develops an automated essay grading system using a fine-tuned instruction-based **Mistral-7B** model. It takes a question, a reference answer, a student answer, and a mark scheme — and returns a **score** and **rationale**.

---

## 🎯 Project Objectives

- ✅ Clean and structure essay grading data
- ✅ Format data for instruction tuning
- ✅ Fine-tune Mistral-7B with LoRA and Unsloth (4-bit)
- ✅ Evaluate predictions vs. human scores
- ✅ Deploy as a web app using Flask + Streamlit

---

## 🏗️ Project Components

| Component       | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `flask_grading_api.py` | Flask backend to query the HF model and serve grades              |
| `essay_grader_fixed.html` | Frontend form to submit answers and display results          |
| `mistral_model_with_tuning.ipynb` | Notebook for fine-tuning using Unsloth & LoRA         |
| `model_with_fine_tuning_results.csv` | CSV of predictions vs. true scores for evaluation |
| `.env`          | Stores Hugging Face token and model name securely                          |

---

## 🧠 Model

- **Name**: [`sue888888888888/essay_grader`](https://huggingface.co/sue888888888888/essay_grader)
- **Base**: `Mistral-7B-Instruct`
- **Quantization**: 4-bit via `bitsandbytes`
- **Fine-Tuning**: LoRA with Unsloth
- **Inputs**: question, reference answer, student answer, mark scheme
- **Output**: total score + explanation

---

## 📊 Evaluation Results

| Metric                    | Value   |
|---------------------------|---------|
| Accuracy                  | 92.5%   |
| MAE (Mean Absolute Error) | 0.075   |
| RMSE                      | 0.27    |
| F1 Score (Macro)          | 0.92    |
| Spearman Rank Correlation | 0.97    |
| Test Samples              | 40      |

---

## 🚀 Run the App

1. **Clone the repo**:
   ```bash
   git clone https://github.com/HalaKhalifa/automatic-essay-grading.git
   cd automatic-essay-grading
   ```
2. **Set up environment**:
   ```bash
    pip install -r requirements.txt
    ```
3. **Create a .env file**:
   ```bash
    HUGGINGFACE_TOKEN=your_hf_token
    ```
4. **Run Flask API**:
   ```bash
    python frontend/flask_grading_api.py
    ```
5. **Access the UI**:
    Visit http://localhost:5000 and test essay grading live.


## 🔗 Resources

🔗 Model on Hugging Face
🔗 Training Dataset
🔗 GitHub Repo

## 👩‍💻 Authors

- **Sama Shalabi**
- **Bissan Dwekat**
- **Hala Khalifeh**
