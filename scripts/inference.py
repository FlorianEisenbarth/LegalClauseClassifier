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

def run_inference(client: Mistral, model: str, messages_list: list) -> list[dict]:
    """
    Performs inference on a list of message samples using a specified Mistral model.

    This function iterates through each sample in the provided list,
    sends the messages (excluding the final ground truth) to the Mistral
    API for a chat completion, and collects the model's predictions.

    Args:
        client (Mistral): An authenticated client object for the Mistral AI API.
        model (str): The identifier of the model to use for inference. This must be
                    a Mitral model ID 
        messages_list (list): A list of dictionaries, where each dictionary
                            represents a single inference sample. Each sample
                            must contain a key 'messages' which holds the
                            conversation history.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              'input' (the messages sent to the model), 'ground_truth'
              (the expected response), and 'prediction' (the model's output).
              If an error occurs, 'prediction' will be None and an 'error'
              key will be included.
    """
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

def main(args):
    """
    Main entry point for inference.
    
    This function orchestrates the entire inference workflow and save
    the prediction into a JSONL file.
    
    Args:
        args: command-line arguments. Expected attributes are:
              - test_file (str): The path to the test dataset in JSONL format.
              - model_id (str): The Mistral ID of the fine-tuned model to use.
              - output_path (str): The path to save the prediction results in JSONL format.

    Raises:
        EnvironmentError: If the 'MISTRAL_API_KEY' environment variable is not set.
    """
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if mistral_key is None:
        raise EnvironmentError("Please set the MISTRAL_API_KEY environment variable.")
    
    client = Mistral(api_key=mistral_key)
    

    all_models = client.models.list()
    model_ids = {m.id for m in all_models.data}
    if args.model_id not in model_ids:
        raise ValueError(f"Model ID '{args.model_id}' is not a valid model.")
    
    test_data = load_jsonl(args.test_file)
    predictions = run_inference(client, model=args.model_id, messages_list=test_data)
    save_jsonl(predictions, args.output_path)

if __name__ == "__main__":
    args = parse_args_inference()
    main(args)