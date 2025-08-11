
import json
import os
import logging
from sklearn.metrics import f1_score
import evaluate
from config import parse_args_eval
import pandas as pd
import numpy as np
from tqdm import trange


def load_data(file_path) -> tuple:
    """
    Loads ground truth and prediction data from a JSONL file.

    Args:
        file_path (str): The path to the input JSONL file.

    Returns:
        tuple: A tuple containing four NumPy arrays in the following order:
            - y_true_clauses (np.array): The ground truth clause types.
            - y_true_summary (np.array): The ground truth summaries.
            - y_pred_clause (np.array): The predicted clause types.
            - y_pred_summary (np.array): The predicted summaries.
    """
    y_true_clauses, y_pred_clause = [], []
    y_true_summary, y_pred_summary = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            
            gt_clause = json.loads(obj["ground_truth"])["clause_type"].strip().lower()
            pred_clause = json.loads(obj["prediction"])["clause_type"].strip().lower()
            gt_summary = json.loads(obj["ground_truth"])["summary"].strip().lower()
            pred_summary = json.loads(obj["prediction"])["summary"].strip().lower()

            y_true_clauses.append(gt_clause)
            y_pred_clause.append(pred_clause)
            y_true_summary.append(gt_summary)
            y_pred_summary.append(pred_summary)
    return np.array(y_true_clauses), np.array(y_true_summary), np.array(y_pred_clause), np.array(y_pred_summary)


def evaluate_summarization(y_true_base: list, 
                            y_true_impr: list, 
                            y_pred_base: list, 
                            y_pred_impr: list,
                            output_file: str) -> None:
    """
    Evaluates and compares the summarization performance between a baseline and an improved model.
    
    This function calculates ROUGE-1, ROUGE-L and BLEU scores to compares the performances beetween
    a base and a finetuned model. The scores are saved in a text file.

    Args:
        y_true_base (list): A list of ground truth labels for the baseline model.
        y_true_impr (list): A list of ground truth labels for the improved model.
        y_pred_base (list): A list of predicted labels from the baseline model.
        y_pred_impr (list): A list of predicted labels from the improved model.
        output_file (str): The path to the directory where the output file
                           'summarization_analysis.txt' will be saved.
    """

    assert np.array_equal(y_true_base, y_true_impr), "Ground truth mismatch between files"
    
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    rouge_results_base = rouge.compute(predictions=y_pred_base, references=y_true_base, use_stemmer=True)
    rouge_results_impr = rouge.compute(predictions=y_pred_impr, references=y_true_impr, use_stemmer=True)
    bleu_results_base = bleu.compute(predictions=y_pred_base, references=y_true_base)
    bleu_results_impr = bleu.compute(predictions=y_pred_impr, references=y_true_impr ) 
    
    os.makedirs(output_file, exist_ok=True)
    with open(os.path.join(output_file, "summarization_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(f"{'Scores':13} | {'Baseline':>8} | {'Improved':>8} | {'Δ score':>8}\n")
        f.write("-"*46+"\n")
        f.write(f"{'Rouge-1':13} | {rouge_results_base['rouge1']:8.3f} | {rouge_results_impr['rouge1']:8.3f} | {rouge_results_impr['rouge1'] - rouge_results_base['rouge1']:8.3f}\n")
        f.write(f"{'Rouge-L':13} | {rouge_results_base['rougeL']:8.3f} | {rouge_results_impr['rougeL']:8.3f} | {rouge_results_impr['rougeL'] - rouge_results_base['rougeL']:8.3f}\n")
        f.write(f"{'BLEU':13} | {bleu_results_base['bleu']:8.3f} | {bleu_results_impr['bleu']:8.3f} | {bleu_results_impr['bleu'] - bleu_results_base['bleu']:8.3f}\n")


def evaluate_classification(y_true_base: list, 
                            y_true_impr: list, 
                            y_pred_base: list, 
                            y_pred_impr: list,
                            output_file: str) -> None:
    """
    Evaluates and compares the classification performance between a baseline and an improved model.

    This function calculates F1-scores (micro, macro, weighted) and performs a
    bootstrap significance test to determine if the improved model is statistically
    superior to the baseline. The detailed analysis is saved to a text file.

    Args:
        y_true_base (list): A list of ground truth labels for the baseline model.
        y_true_impr (list): A list of ground truth labels for the improved model.
        y_pred_base (list): A list of predicted labels from the baseline model.
        y_pred_impr (list): A list of predicted labels from the improved model.
        output_file (str): The path to the directory where the output file
                           'classification_analysis.txt' will be saved.

    Raises:
        AssertionError: If the ground truth labels for the two models do not match.

    """
    labels = list(set(y_true_base))

  
    assert np.array_equal(y_true_base, y_true_impr), "Ground truth mismatch between files"

    os.makedirs(output_file, exist_ok=True)
    with open(os.path.join(output_file, "classification_analysis.txt"), "w", encoding="utf-8") as f:


        f.write("==============================\n")
        f.write("Overall F1 Scores\n")
        f.write("==============================\n")

        f1_base_macro = f1_score(y_true_base, y_pred_base, average="macro", labels=labels, zero_division=0)
        f1_impr_macro = f1_score(y_true_impr, y_pred_impr, average="macro", labels=labels, zero_division=0)

        f1_base_micro = f1_score(y_true_base, y_pred_base, average="micro", labels=labels, zero_division=0)
        f1_impr_micro = f1_score(y_true_impr, y_pred_impr, average="micro", labels=labels, zero_division=0)

        f1_base_weighted = f1_score(y_true_base, y_pred_base, average="weighted", labels=labels, zero_division=0)
        f1_impr_weighted = f1_score(y_true_impr, y_pred_impr, average="weighted", labels=labels, zero_division=0)

        f.write(f"{'F1 scores':13} | {'Baseline':>8} | {'Improved':>8} | {'Δ F1':>8}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'F1-micro':13} | {f1_base_micro:8.3f} | {f1_impr_micro:8.3f} | {f1_impr_micro - f1_base_micro:8.3f}\n")
        f.write(f"{'F1-macro':13} | {f1_base_macro:8.3f} | {f1_impr_macro:8.3f} | {f1_impr_macro - f1_base_macro:8.3f}\n")
        f.write(f"{'F1-weighted':13} | {f1_base_weighted:8.3f} | {f1_impr_weighted:8.3f} | {f1_impr_weighted - f1_base_weighted:8.3f}\n\n")


        f.write("==============================\n")
        f.write("Bootstrap Significance Test\n")
        f.write("==============================\n")

        n = len(y_true_base)
        diffs = []
        n_bootstrap = 1000 
        rng = np.random.default_rng(42) 

        for _ in trange(n_bootstrap, desc="Bootstrapping on micro-F1"):
            idx = rng.integers(0, n, n)
            f1_b = f1_score(y_true_base[idx], y_pred_base[idx], average="micro", labels=labels, zero_division=0)
            f1_i = f1_score(y_true_impr[idx], y_pred_impr[idx], average="micro", labels=labels, zero_division=0)
            diffs.append(f1_i - f1_b)

        diffs = np.array(diffs)
      
        p_value = (np.sum(diffs <= 0) + 1) / (n_bootstrap + 1)

        ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])

        f.write(f"Bootstrap 95% CI for improvement: [{ci_lower:.3f}, {ci_upper:.3f}]\n")
        f.write(f"One-sided p-value (improved > baseline): {p_value:.6f}\n\n")

        f.write("==============================\n")
        f.write("Per-Class F1 Changes\n")
        f.write("==============================\n")

        classes = sorted(list(set(y_true_base)))
        per_class_results = []

        for cls in classes:
            mask = (y_true_base == cls)
            f1_b = f1_score(y_true_base[mask], y_pred_base[mask], average="micro", labels=labels, zero_division=0)
            f1_i = f1_score(y_true_impr[mask], y_pred_impr[mask], average="micro", labels=labels, zero_division=0)
            per_class_results.append((cls, f1_b, f1_i, f1_i - f1_b))

        per_class_results.sort(key=lambda x: x[3], reverse=True)

        f.write(f"{'Class':50} | {'Baseline':>8} | {'Improved':>8} | {'Δ F1':>8}\n")
        f.write("-" * 85 + "\n")
        for cls, f1_b, f1_i, delta in per_class_results:
            f.write(f"{cls:50} | {f1_b:8.3f} | {f1_i:8.3f} | {delta:8.3f}\n")

    print(f"Analysis complete. Results have been saved to '{output_file}'.")
    


def main(args):
     
    try:
        y_true_base_clause, y_true_base_summary, y_pred_base_clause, y_pred_base_summary = load_data(args.references_path)
        y_true_impr_clause, y_true_impr_summary, y_pred_impr_clause, y_pred_impr_summary = load_data(args.predictions_path)
    except FileNotFoundError as e:
        print(f"Error: One of the prediction files was not found. Please check the paths.")
        print(e)
        exit()
        
    evaluate_classification(y_true_base= y_true_base_clause,
                            y_pred_base=y_pred_base_clause,
                            y_true_impr=y_true_impr_clause,
                            y_pred_impr=y_pred_impr_clause,
                            output_file=args.output_path)
    
    evaluate_summarization(y_true_base= y_true_base_summary,
                           y_pred_base=y_pred_base_summary,
                           y_true_impr=y_true_impr_summary,
                           y_pred_impr=y_pred_impr_summary,
                           output_file=args.output_path
    )


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    args = parse_args_eval()
    main(args)