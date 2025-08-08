import os
import json
import jsonlines
from tqdm import tqdm
from mistralai import Mistral
from config import parse_args_inference
import logging
from utils import save_jsonl, load_jsonl
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_inference(client: Mistral, model: str, messages_list: list):
    """Perform inference on a list of messages using the provided model."""
    outputs = []
    for sample in tqdm(messages_list, desc="Running inference", position=0, leave=True):
        try:
            response = client.chat.complete(
                model=model,
                messages=sample["messages"][:-1],
                response_format={"type": "json_object"}
            )
            outputs.append({
                "input": sample["messages"][:-1],
                "ground_truth": sample["messages"][-1]["content"],
                "prediction": response.choices[0].message.content
            })
        except Exception as e:
            logging.error(f"Inference failed for sample: {sample} | Error: {e}")
            outputs.append({
                "input": sample["messages"][:-1],
                "ground_truth": sample["messages"][-1]["content"],
                "prediction": None,
                "error": str(e)
            })
    return outputs

def inference_finetuned(args):
    """Main entry point for inference."""
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if mistral_key is None:
        raise EnvironmentError("Please set the MISTRAL_API_KEY environment variable.")
    
    client = Mistral(api_key=mistral_key)
    
    test_data = load_jsonl(args.test_file)
    predictions = run_inference(client, model=args.model_id, messages_list=test_data)
    save_jsonl(predictions, args.output_path)

if __name__ == "__main__":
    args = parse_args_inference()
    inference_finetuned(args)