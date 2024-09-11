import json
from dataclasses import asdict, dataclass
from datetime import datetime

import boto3
import torch
from aws_lambda_powertools.logging import Logger
from dynamodbinterface import DynamoDBInterface, StorableDynamoDB
from models import FramingModel
from transformers import AutoConfig


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

# Define the list of labels
labels_list = [
    "Economic",
    "Capacity_and_resources",
    "Morality",
    "Fairness_and_equality",
    "Legality_Constitutionality_and_jurisprudence",
    "Policy_prescription_and_evaluation",
    "Crime_and_punishment",
    "Security_and_defense",
    "Health_and_safety",
    "Quality_of_life",
    "Cultural_identity",
    "Public_opinion",
    "Political",
    "External_regulation_and_reputation",
]


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

    # Load the custom model
    xlmrobertaconfig = AutoConfig.from_pretrained("./models/xlm-roberta-base")
    framingmodel = FramingModel(xlmrobertaconfig)
    logger.info("Model initialized")

    # Load the model state dict
    ckpt = torch.load("./models/subtask2.pt", map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    for k in list(ckpt.keys()):
        if k.startswith("head2."):
            # rename the key to classifier
            ckpt[k.replace("head2.", "classifier.")] = ckpt.pop(k)
        if k.startswith("model."):
            # rename the key to classifier
            ckpt[k.replace("model.", "")] = ckpt.pop(k)
    framingmodel.load_state_dict(ckpt, strict=False)
    logger.info("Model state dict loaded")

    # Set the model to evaluation mode
    framingmodel.eval()
    logger.info("Model set to evaluation mode")

    # Get the labels list
    outputs = framingmodel(text)

    # Get the probabilities
    probabilities = torch.sigmoid(outputs)
    probabilities = probabilities.flatten().tolist()
    logger.info(f"Probabilities: {probabilities}")

    # Filter the labels and outputs
    output_map = {}
    filtered_labels = []
    filtered_outputs = []
    for index, output in enumerate(probabilities):
        if labels_list[index] == "None":
            continue
        if output >= 0.35:
            filtered_labels.append(labels_list[index])
            filtered_outputs.append(output)
    logger.info(f"Filtered labels: {filtered_labels}")

    # Store the filtered labels and outputs in the map with the corresponding sentence
    output_map = {"Labels": filtered_labels, "Probabilities": filtered_outputs}

    # Update the DynamoDB table
    response["status"] = "COMPLETED"
    ddbi.create_or_update(
        Entry(
            id=response["id"],
            payload=response["payload"],
            status="COMPLETED",
            task=response["task"],
            response=json.dumps(output_map),
            last_updated=str(datetime.now()),
        )
    )

    return {"statusCode": 200, "body": json.dumps(output_map)}


if __name__ == "__main__":
    event = {
        "task_id": "c059ae30b28e45cf9eb45c906dd5b578",
    }
    context = {}
    lambda_handler(event, context)
