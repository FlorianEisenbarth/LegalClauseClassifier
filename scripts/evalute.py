
import json
import os
import logging
from tqdm import tqdm
from utils import load_jsonl
from sklearn.metrics import classification_report
import evaluate
from config import parse_args_eval
import pandas as pd


def format_prediction(prediction: list, references: list):
    """Formats prediction and reference data to facilitate evaluation."""
    format_output = []

    for pred, ref in zip(prediction, references):
        print(type(pred))
        try:
            pred_json = json.loads(pred['prediction'])
            pred_label = pred_json.get("clause_type", "Unknown")
        except json.JSONDecodeError:
            pred_label = "Unknown"

        message = ref["messages"][-1]["content"]
        try:
            true_label = json.loads(message).get("clause_type", "Unknown")
        except Exception:
            true_label = "Unknown"
        try:
            pred_json = json.loads(pred['prediction'])
            pred_summary = pred_json.get("summary", "")
        except json.JSONDecodeError:
            pred_summary = ""
        message = ref["messages"][-1]["content"]
        try:
            ref_summary = json.loads(message).get("summary", "")
        except Exception:
            ref_summary = ""
        
        format_output.append({
            "predicted_clause_type": pred_label,
            "reference_clause_type": true_label,
            "predicted_summary": pred_summary,
            "reference_summary": ref_summary
        })
    return format_output

def evaluate_clause_type(predictions: list):
    """Produce classification report on clause types"""
    
    y_true = [pred["reference_clause_type"] for pred in predictions]
    y_pred = [pred["predicted_clause_type"] for pred in predictions]
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True, labels=list(set(y_true)))
    logging.info("\n=== Clause Type Classification Report ===")
    logging.info(pd.DataFrame(report).T)
    logging.info("\n=== F1 micro average ===")
    if "accuracy" in report.keys():
        logging.info(report.get("accuracy"))
    else:
        logging.info(report.get('micro avg')["f1-score"])
    return pd.DataFrame(report).T


def evaluate_summary(predictions: list):
    """Evaluates the quality of a list of predicted summaries against their reference summaries using ROUGE/BLEU score"""
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    references = [pred["reference_summary"] for pred in predictions]
    predictions_text = [pred["predicted_summary"] for pred in predictions]

    
    rouge_results = rouge.compute(predictions=predictions_text, references=references, use_stemmer=True)
    
    
    bleu_results = bleu.compute(predictions=predictions_text,
                                references=references)
    scores = {
        "rouge1": rouge_results["rouge1"],
        "rougeL": rouge_results["rougeL"],
        "bleu": bleu_results["bleu"]
    }
    logging.info("=== Summary Evaluation ===")
    logging.info(scores)
    
    return scores
  
def main(args):
    

    if not os.path.exists(args.predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {args.predictions_path}")
    if not os.path.exists(args.references_path):
        raise FileNotFoundError(f"References file not found: {args.references_path}")

    # Load predictions
    predictions =  load_jsonl(args.predictions_path)

    # Load test file
    references = load_jsonl(args.references_path)
    predictions_formated = format_prediction(predictions, references)
    
    # perform evaluation
    report = evaluate_clause_type(predictions_formated)
    rouge_scores = evaluate_summary(predictions_formated)
    
    #save the evaluation
    os.makedirs(args.output_path, exist_ok=True)
    report_path = os.path.join(args.output_path, "classification_report.txt")
    report.to_string(report_path)
    rouge_scores_path = os.path.join(args.output_path, "rouges_scores.txt")
    with open(rouge_scores_path,"w") as f:
        f.writelines(str(rouge_scores))
    


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    args = parse_args_eval()
    main(args)