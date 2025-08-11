from mistralai import Mistral
import os
from dotenv import load_dotenv
import json
import time
from config import parse_args_finetuning


load_dotenv()

MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

def main(args):
    """
    Initializes and monitors a fine-tuning job on the Mistral AI La Platforme.
    
    Args:
        args: command-line arguments, expected to have the following attributes:
            - train_path (str): The local file path to the training dataset.
            - eval_path (str): The local file path to the evaluation dataset.
            - base_model_finetuning (str): The identifier for the base model to be fine-tuned.

    Raises:
        FileNotFoundError: If the specified training or evaluation files do not exist.
        
    Notes:
        - The function requires the 'MISTRAL_API_KEY' environment variable to be set for authentication.
    """
    
    
    client = Mistral(api_key=MISTRAL_API_KEY)

    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"train file not found: {args.train_path}")
    if not os.path.exists(args.eval_path):
        raise FileNotFoundError(f"eval file not found: {args.eval_path}")


    cuad_classification_summary_train = client.files.upload(file={
        "file_name": os.path.split(args.train_path)[-1],
        "content": open(args.train_path, "rb"),
    })
    cuad_classification_summary_eval = client.files.upload(file={
        "file_name": os.path.split(args.eval_path)[-1],
        "content": open(args.eval_path, "rb"),
    })


    created_job = client.fine_tuning.jobs.create(
        model=args.base_model_finetuning, 
        training_files=[{"file_id": cuad_classification_summary_train.id, "weight": 1}],
        validation_files=[cuad_classification_summary_eval.id],
        hyperparameters={
            "epochs": 2,
            "learning_rate":2e-4,
        },
        auto_start=True,
    )

    retrieved_job = client.fine_tuning.jobs.get(job_id=created_job.id)

    while retrieved_job.status in ["QUEUED", "RUNNING", "VALIDATING"]:
        retrieved_job = client.fine_tuning.jobs.get(job_id=created_job.id)
        job_info = json.dumps(retrieved_job.model_dump(), indent=4)
        if len(job_info) > 10000:
            print(job_info[:5000] + "\n[...]\n" + job_info[-5000:])
        else:
            print(job_info)
        time.sleep(5)
    print(retrieved_job.status)
    if retrieved_job.status == "SUCCESS":
        print("Finetung status: SUCCESS")
        print(f"model id: {retrieved_job.fine_tuned_model}")

if __name__ == "__main__":
    
    args = parse_args_finetuning()
    main(args)
