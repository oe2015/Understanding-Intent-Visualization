import json
from dataclasses import asdict, dataclass
from datetime import datetime

import boto3
import torch
from aws_lambda_powertools.logging import Logger
from dynamodbinterface import DynamoDBInterface, StorableDynamoDB
from models import CustomModel
from transformers import AutoConfig, AutoTokenizer


@dataclass
class Entry(StorableDynamoDB):
    id: int
    task: str
    status: str
    payload: dict
    response: dict
    last_updated: str

    def items(self):
        """Returns the items of the dictionary representing the data class."""
        return asdict(self).items()

    def PK(self):
        return f"id#{self.id}"


# Initialize the logger
logger = Logger()


def lambda_handler(event, context):
    # Log the event
    logger.info(f"Received event: {event}")
    logger.info(f"Received context: {context}")

    # Get the text from the request
    task_id = event["task_id"]

    # Query the entry from the database
    ddbi = DynamoDBInterface(
        dynamodb_svc=boto3.client("dynamodb", region_name="us-east-1"),
        table="frappe-db-devhasaniqbal-us-east-1",
    )
    response = ddbi.get(f"id#{task_id}")

    # Get the payload from the response
    payload_str = response.get("payload")
    if payload_str is None:
        raise Exception("Payload not found in the database")

    # Parse the payload
    payload = json.loads(payload_str)
    text = payload.get("text")
    if text is None:
        raise Exception("Text not found in the payload")

    # Load the models and tokenizers
    tokenizer = AutoTokenizer.from_pretrained("./models/xlm-roberta-base")
    logger.info("Tokenizers loaded")

    # Load the configuration and initialize the model
    custom_model = CustomModel()
    logger.info("Model initialized")

    # Load and adjust the model state dict
    state_dict = torch.load("./models/best_model-v1.ckpt", map_location="cpu")["state_dict"]
    adjusted_state_dict = {
        k.replace("model.model.", "model.").replace("model.fc.", "fc."): v
        for k, v in state_dict.items()
        if "model.embeddings.position_ids" not in k
    }
    custom_model.load_state_dict(adjusted_state_dict)
    logger.info("Model state dict loaded")

    # Set the model to evaluation mode
    custom_model.eval()
    logger.info("Model set to evaluation mode")

    # Get the labels list
    inputs = tokenizer(
        text,
        max_length=512,
        padding="longest",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    outputs = custom_model(inputs)

    # Get the probabilities
    probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()
    logger.info(f"Probabilities: {probabilities}")

    # Update the DynamoDB table
    ddbi.create_or_update(
        Entry(
            id=response["id"],
            payload=response["payload"],
            status="COMPLETED",
            task=response["task"],
            response=json.dumps({"Probabilities": probabilities}),
            last_updated=str(datetime.now()),
        )
    )

    # Return the predicted probabilities
    return {"statusCode": 200, "body": json.dumps({"Probabilities": probabilities})}
