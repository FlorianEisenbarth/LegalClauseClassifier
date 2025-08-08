# utils.py

import os
import json
import jsonlines
import logging


def load_jsonl(data_path: str):
    """Load test prompts/messages from a JSONL file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data file not found at: {data_path}")
    
    with open(data_path, "r") as f:
        return [json.loads(line.strip()) for line in f]

def save_jsonl(data: list, output_path: str):
    """Save predictions to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(data)
    logging.info(f"Saved file to {output_path}")