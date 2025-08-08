# Legal Clause Extraction & Summarization with Mistral

This take-home assignment demonstrates how to build and fine-tune a Large Language Model (LLM) for **legal clause classification and summarization** from contract documents using the [CUAD dataset](https://github.com/TheAtticusProject/cuad).

The goal is to:
1. Classify the clause type (e.g., Confidentiality, Termination, etc.).
2. Summarize each clause in plain English.

The model is fine-tuned using Mistral's API and evaluated using classification and summarization metrics (Precision, Recall, F1, ROUGE, BLEU).

---

## Setup Environment

### 1. Environment

You can create a virtual environment using conda:

```bash
conda env create -f environment.yml
conda activate LegalClassifier
```

### 2. env file

create a .env file and place you Mistral API key inside
```bash
MISTRAL_API_KEY = <your_api_key>
```

## How to run

### 1. Data Preparation
(This step involves calling up the mistral-large model; for the simplicity, I've included the dataset in the repository).
```bash 
python scripts/prepare_dataset.py
```
### 2. inference with base model
```bash
python scripts/inference.py \
    --model_id "ministral-8b-2410" \
    --output_path "outputs/base_output.jsonl" \
    --test_file "dataset/cuad_classification_summary_test_balanced.jsonl"
```
### 3. Fine-tune mistral model
```bash
python scripts/finetune_mistral.py --base_model_finetuning ministral-3b-latest            
```
When the finetuning is over the script will output the finetuned model id
### 4. inference with finetuned model
```bash
scripts/inference.py \
    --model_id <model id> \
    --output_path "outputs/finetuned_output.jsonl" \
    --test_file "dataset/cuad_classification_summary_test_balanced.jsonl"  
```
### 5. Evaluation
```bash
python scripts/evalute.py \
    --predictions_path "outputs/base_output.jsonl" \
    --references_path "dataset/cuad_classification_summary_test_balanced.jsonl" \
    --output_path "outputs/eval/base_model"

 python scripts/evalute.py \
    --predictions_path "outputs/finetuned_output.jsonl" \
    --references_path "dataset/cuad_classification_summary_test_balanced.jsonl" \
    --output_path "outputs/eval/finetuned_model"     

```

## Evaluation Results
I decided to push the <code>outputs</code> folder In case you would not have the time to run the full pipeline.
### Classification Task Evaluation
The primary goal of this project was to fine-tune a model to accurately classify legal clauses into one of 41 specific types. We compared the performance of our Fine-Tuned Model against a Base Model that struggled with the task, often outputting predictions that did not correspond to any of the 41 valid clause types.

The evaluation was performed on a test dataset with a total of 1664 samples. The results are summarized below.

Key Findings
The fine-tuned model demonstrated a significant improvement in performance across all key metrics when compared to the base model.
| Metric         |  Fine-tuned (ministral-3b-latest) |Base Model (ministral-8b-2410) | Analysis | 
| ---------------| ------------------------------|-----------------------------------|----------|
| Accuracy       |              **0.70**         |     0.55 (micro avg)              | The fine-tuned model correctly classified nearly 70% of all samples, a 15% increase over the base model
| Macro avg F1   |              **0.62**         |     0.48                          |This metric gives equal weight to each of the 41 clause types. The fine-tuned model's higher score indicates it performs better, on average, at identifying all different types of clauses, including minority classes.
| Weighted Avg F1|              **0.68**         |     0.51                          | This metric weights each class by its support (number of samples). The fine-tuned model's higher score shows it is more effective at classifying the most frequent clause types.

#### Anomaly in Base Model Metrics

An important observation was the discrepancy between the base model's micro-averaged precision, recall, and F1-score. In a standard multi-class classification problem, these three values are always identical and equal to the overall accuracy.

* Base Model Micro Avg: Precision (0.546), Recall (0.51), F1 (0.53)

* Fine-Tuned Model Accuracy/Micro Avg: 0.70 for all three metrics

The differing values for the base model's micro-average metrics are a direct result of it predicting "wrong labels outside of the 41 clauses types." These invalid predictions create a situation where the total number of false positives is not equal to the total number of false negatives, breaking the mathematical equivalence that normally exists between these metrics. This highlights the base model's fundamental failure to adhere to the defined classification task.


#### Full per-category classification report (all 41 clause types) is available here:
<code> output/eval/base_model/classification_report.txt** </code>
<code>output/eval/finetuned_model/classification_report.txt </code>

### Summarization Task Evaluation
I also evaluated the models on a text summarization task. ROUGE scores measure the overlap of n-grams between the generated and reference summaries, while BLEU measures the fluency and accuracy of the generated text.
| Metric         | Fine-tuned (ministral-3b-latest) | Base Model (ministral-8b-2410) | Analysis | 
| ---------------| ------------------------------|-----------------------------------|----------|
| ROUGE-1        |           0.62                |    0.54                           |  greater overlap of unigrams (single words) with the reference summaries.
| ROUGE-L        |           0.55                |    0.47                           |  the fine-tuned model generates more fluent and coherent sentences.
| BLEU           |           0.29                |    0.16                           | The fine-tuned model's higher BLEU score reflects its ability to produce summaries that are both more accurate and grammatically correct.

## Streamlit UI
### How to run

Add your fine-tuned a <code>model_id</code> inside your .env file and place you Mistral API key inside
```bash
 FINE_TUNED_MODEL_ID= <model_id>
```
Then launch the app with the following command
```bash
streamlit run app.py
```

