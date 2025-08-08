import argparse


def parse_args_inference():
    
    parser = argparse.ArgumentParser(description="LegalClauseExtractor_inference")
    
    parser.add_argument("--model_id", type=str, default='ministral-3b-latest', help="model id to be used for inference")
    parser.add_argument("--output_path", type=str, default="outputs/base_outputs.jsonl")
    parser.add_argument("--test_file", type=str, default="dataset/cuad_classification_summary_test_balanced.jsonl")
    args = parser.parse_args()
    
    return args


def parse_args_eval():
    
    parser = argparse.ArgumentParser(description="LegalClauseExtractor_eval")
    
    parser.add_argument("--predictions_path", type=str, default='outputs/base_outputs.jsonl', help="Path to the model predictions (jsonl format).")
    parser.add_argument("--references_path", type=str, default="dataset/cuad_classification_summary_test_balanced.jsonl",   help="Path to the reference test data (jsonl format).")
    parser.add_argument("--output_path", type=str, default="outputs/base_outputs_eval", help="Path where the classification report and ROUGE/BLEU scores will be saved")
    args = parser.parse_args()
    
    return args



def parse_args_finetuning():
    
    parser = argparse.ArgumentParser(description="LegalClauseExtractor_finetuning")
    
    parser.add_argument("--base_model_finetuning", type=str, default='ministral-3b-latest', help="id of the model to be finetuned")
    parser.add_argument("--train_path", type=str, default="dataset/cuad_classification_summary_train_balanced.jsonl",   help="Path to the train data (jsonl format).")
    parser.add_argument("--eval_path", type=str, default="dataset/cuad_classification_summary_eval_balanced.jsonl",   help="Path to the eval data (jsonl format).")
    args = parser.parse_args()
    
    return args