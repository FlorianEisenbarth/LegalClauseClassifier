import json
import jsonlines
import pandas as pd 
from sklearn.model_selection import train_test_split
from mistralai import Mistral
import os
from tqdm import tqdm
from collections import defaultdict
from dotenv import load_dotenv
import random
from utils import save_jsonl
from prompts import PROMPTS


load_dotenv()

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=MISTRAL_API_KEY)

with open("data/CUAD_v1/CUAD_v1.json") as f:
    cuda_v1 = json.load(f)
    

def prepare_finetuning_dataset_classification_balanced():
    """Prepares a balanced dataset for a classification task from the CUAD dataset."""

    samples_by_clause = defaultdict(list)

    MAX_SAMPLES_PER_CLAUSE = 300

    for document in tqdm(cuda_v1['data'], desc="Generating the dataset..."):
        for paragraph in document['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                if qa["is_impossible"] or not qa["answers"]:
                    continue
                clause_name = qa["id"].split("__")[-1].strip()
                if clause_name == "Rofr/Rofo/Rofn":
                    clause_name = "Right of First Refusal, Offer or Negotiation"
                for answer in qa['answers']:

                    answer_text = answer["text"].strip()

                    if len(answer_text) < 3:
                        continue

                    start = answer["answer_start"]
                    end = start + len(answer_text)
                    paragraph = context[max(0, start):min(len(context), end)]
                    if len(samples_by_clause[clause_name]) <= MAX_SAMPLES_PER_CLAUSE:
                        messages = [
                    {
                        "role": "system",
                        "content": PROMPTS['system_classification']
                    },
                    {
                        "role": "user",
                        "content": f"Paragraph: {paragraph}"
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"clause_type": clause_name.lower()})
                    }
                ]
                        samples_by_clause[clause_name].append({'messages':messages})

    all_samples = []
    for clause_samples in samples_by_clause.values():
            all_samples.extend(clause_samples)

    random.seed(42)
    random.shuffle(all_samples)

    
    train_data_balanced, test_data_balanced = train_test_split(all_samples, test_size=0.2, random_state=42)
    train_data_balanced, eval_data_balanced = train_test_split(train_data_balanced, test_size=0.2, random_state=42)

   
    save_jsonl(train_data_balanced,"dataset/cuad_classification_train_balanced.jsonl")
    save_jsonl(eval_data_balanced,"dataset/cuad_classification_eval_balanced.jsonl")
    save_jsonl(test_data_balanced,"dataset/cuad_classification_test_balanced.jsonl")


    
def prepare_dataset_classification_sumarization_balanced():
    """Prepares a balanced dataset for a combined classification and summarization task from the CUAD dataset."""

    samples_by_clause = defaultdict(list)

    MAX_SAMPLES_PER_CLAUSE = 300

    for document in tqdm(cuda_v1['data'], desc="Generating the dataset..."):
        for paragraph in document['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                if qa["is_impossible"] or not qa["answers"]:
                    continue
                clause_name = qa["id"].split("__")[-1].strip()
                if clause_name == "Rofr/Rofo/Rofn":
                    clause_name = "Right of First Refusal, Offer or Negotiation"
                for answer in qa['answers']:

                    answer_text = answer["text"].strip()

                    if len(answer_text) < 3:
                        continue

                    start = answer["answer_start"]
                    end = start + len(answer_text)
                    paragraph = context[max(0, start):min(len(context), end)]
                    if len(samples_by_clause[clause_name]) <= MAX_SAMPLES_PER_CLAUSE:
                        # Generate summary from Mistral
                        output = client.chat.complete(
                            model="mistral-large-2411",
                            messages=[
                                {'role': "system", "content": "You are a legal AI assistant. You will receive a clause paragraph from a legal contract and your task is to summarize it using simple terms anyone can understand. Output only the summary."},
                                {"role": "user", "content": f"Clause paragraph:\n{paragraph}\nSummary:"}
                            ]
                        )
                        summary = output.choices[0].message.content.strip()

                        messages = [
                    {
                        "role": "system",
                        "content": PROMPTS['system_classification_summary']
                    },
                    {
                        "role": "user",
                        "content": f"Paragraph: {paragraph}"
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"clause_type": clause_name.lower(), "summary": summary})
                    }
                ]
                        samples_by_clause[clause_name].append({'messages':messages})

    all_samples = []
    for clause_samples in samples_by_clause.values():
            all_samples.extend(clause_samples)

    random.seed(42)
    random.shuffle(all_samples)

    
    train_data_balanced, test_data_balanced = train_test_split(all_samples, test_size=0.2, random_state=42)
    train_data_balanced, eval_data_balanced = train_test_split(train_data_balanced, test_size=0.2, random_state=42)

   
    save_jsonl(train_data_balanced,"dataset/cuad_classification_summary_train_balanced.jsonl")
    save_jsonl(eval_data_balanced,"dataset/cuad_classification_summary_eval_balanced.jsonl")
    save_jsonl(test_data_balanced,"dataset/cuad_classification_summary_test_balanced.jsonl")

if __name__ == "__main__":
    
    prepare_dataset_classification_sumarization_balanced()