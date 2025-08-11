# utils.py

import os
import json
import jsonlines
import logging


def load_jsonl(data_path: str) -> list:
    """
    Load test prompts/messages from a JSONL file.
    
    Args:
        data_path (str): The complete file path to the JSONL file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a JSON object
              from a line in the file.

    Raises:
        FileNotFoundError: If the file specified by `data_path` does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data file not found at: {data_path}")
    
    with open(data_path, "r") as f:
        return [json.loads(line.strip()) for line in f]

def save_jsonl(data: list, output_path: str):
    """
    Save predictions to a JSONL file.
    
    Args:
        data (list): list of dictonaries to be saved in a JSONL file.
        output_path (str): The path to the directory where the output file
                        will be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(data)
    logging.info(f"Saved file to {output_path}")