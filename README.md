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

create a .env file at the root of the project and place you Mistral API key inside
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
python scripts/finetune_mistral.py \
    --base_model_finetuning "ministral-3b-latest" \
    --train_path "dataset/cuad_classification_summary_train_balanced.jsonl" \
    --eval_path "dataset/cuad_classification_summary_eval_balanced.jsonl"
```
When the finetuning is over the script will output the finetuned model id
### 4. inference with finetuned model
```bash
python scripts/inference.py \
    --model_id <model id> \
    --output_path "outputs/finetuned_output.jsonl" \
    --test_file "dataset/cuad_classification_summary_test_balanced.jsonl"  
```
### 5. Evaluation
(if you haven't run the pipeline, you can use <code>outputs/base_balanced_classification_summary.json</code> and <code>outputs/finetuned_balanced_classification_summary.jsonl</code> to run the evaluation)
```bash
python scripts/evaluation.py \
    --predictions_path "outputs/finetuned_output.jsonl" \
    --references_path "outputs/base_output.jsonl" \
    --output_path "outputs/eval/"

```

## Evaluation Results
I decided to push the <code>outputs</code> folder In case you would not have the time to run the full pipeline.
### Classification Task Evaluation
The primary goal of this project was to fine-tune a model to accurately classify legal clauses into one of 41 specific types. We compared the performance of our Fine-Tuned Model against a Base Model that struggled with the task, often outputting predictions that did not correspond to any of the 41 valid clause types.

The evaluation was performed on a test dataset with a total of 1664 samples.

The fine-tuned model demonstrated a significant improvement in performance across all key metrics when compared to the base model.

|Metric      | Base Model (ministral-8b-2410) | Fine-tuned (ministral-3b-latest) |     Δ F1    |
-------------|-------------------------------|-----------------------------------|-------------|
|F1-micro    |          0.53                 |             **0.70**              |    0.17
|F1-macro    |          0.48                 |             **0.62**              |    0.14
|F1-weighted |          0.51                 |            **0.68**               |    0.17 



- **Significant overall improvement**:  
  The micro F1 score increased by approximately 0.14 to 0.19 (95% confidence interval), with a statistically significant p-value (0.0001), confirming that fine-tuning meaningfully enhances classification accuracy.

- **Strong per-class gains**:  
  Major improvements were observed on key clause types such as *document name* (+0.73 F1), *covenant not to sue* (+0.71 F1), and *revenue/profit sharing* (+0.63 F1). This demonstrates the model’s improved ability to correctly identify legal clauses.

- **Consistent improvements across many categories**:  
  The majority of clause types showed positive gains in F1 scores, indicating a broad enhancement of model performance rather than isolated improvements.

- **Stable performance on some categories**:  
  Certain clause types retained similar performance levels after fine-tuning, reflecting either a ceiling effect or limited training data variation.

- **Performance drops in a few categories**:  
  A small number of categories, including *license grant* and *price restrictions*, experienced decreased F1 scores, highlighting areas where additional data, data augmentation, or model tuning, changes may be required.


#### Full per-category classification report (all 41 clause types) is available here:
<code> output/eval/base_model/classification_analysis.txt** </code>

### Summarization Task Evaluation
I also evaluated the models on a text summarization task. 
ROUGE scores measure the overlap of n-grams between the generated and reference summaries, while BLEU measures the fluency and accuracy of the generated text.
| Metric         | Base Model (ministral-8b-2410) | Fine-tuned (ministral-3b-latest) |  Δ Metric    | 
| ---------------| ------------------------------|-----------------------------------|--------------|
| ROUGE-1        |           0.54                |    **0.63**                       | 0.09
| ROUGE-L        |           0.47                |    **0.56**                       |  0.08
| BLEU           |           0.17                |    **0.30**                       |  0.13

- **Consistent gains across all metrics**: The fine-tuned model outperforms the baseline on ROUGE-1, ROUGE-L, and BLEU, suggesting it produces summaries that are both more lexically similar to the references and better aligned with their structure.

- **BLEU improvement** is the most notable (+0.129, +74% relative), indicating the fine-tuned model generates more precise n-gram matches.

- **ROUGE-1 and ROUGE-L improvements** (~+0.086–0.087):  better recall and improved ability to capture key terms and longer matching sequences, which is important in legal text where preserving critic-al wording is essential.

The balanced improvements suggest the fine-tuned model is not just memorizing common phrases but has learned domain-specific summarization patterns for legal clauses.


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

