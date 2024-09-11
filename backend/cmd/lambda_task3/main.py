import json
from dataclasses import asdict, dataclass
from datetime import datetime

import boto3
import nltk
import torch
from aws_lambda_powertools.logging import Logger
from dynamodbinterface import DynamoDBInterface, StorableDynamoDB
from models import PersuasionModel
from transformers import AutoConfig, AutoTokenizer
from utils import extract_title_and_sentences


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
    "Appeal_to_Authority",
    "Appeal_to_Popularity",
    "Appeal_to_Values",
    "Appeal_to_Fear-Prejudice",
    "Flag_Waving",
    "Causal_Oversimplification",
    "False_Dilemma-No_Choice",
    "Consequential_Oversimplification",
    "Straw_Man",
    "Red_Herring",
    "Whataboutism",
    "Slogans",
    "Appeal_to_Time",
    "Conversation_Killer",
    "Loaded_Language",
    "Repetition",
    "Exaggeration-Minimisation",
    "Obfuscation-Vagueness-Confusion",
    "Name_Calling-Labeling",
    "Doubt",
    "Guilt_by_Association",
    "Appeal_to_Hypocrisy",
    "Questioning_the_Reputation",
    "None",
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

    # Load the models
    nltk.download("punkt_tab", download_dir="/tmp/nltk_data/")
    nltk.data.path.append("/tmp/nltk_data/")
    logger.info("NLTK punkt tokenizer downloaded")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./models/xlm-roberta-large")

    # Load the custom model
    xlmrobertaconfig = AutoConfig.from_pretrained("xlm-roberta-large")
    persuasion_model = PersuasionModel(xlmrobertaconfig)
    logger.info("Model initialized")

    # Load the model state dict
    ckpt = torch.load("./models/subtask3.pth", map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    for k in list(ckpt.keys()):
        if k.startswith("head2."):
            # rename the key to classifier
            ckpt[k.replace("head2.", "classifier.")] = ckpt.pop(k)
        if k.startswith("model."):
            # rename the key to classifier
            ckpt[k.replace("model.", "")] = ckpt.pop(k)
    persuasion_model.load_state_dict(ckpt, strict=False)
    logger.info("Model state dict loaded")

    # Set the model to evaluation mode
    persuasion_model.eval()
    logger.info("Model set to evaluation mode")

    # Get the labels list
    output_map = {}
    sentences = extract_title_and_sentences(text)
    for sentence in sentences:
        # Tokenize the sentence
        inputs = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=512,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs_3 = {
            "content_input_ids": inputs["input_ids"],
            "content_attention_mask": inputs["attention_mask"],
        }
        # Pass the tokenized sentence to the model
        outputs_3 = persuasion_model(**inputs_3)
        probabilities_3 = torch.sigmoid(outputs_3)
        # probabilities_3 = probabilities_3.tolist()  # Convert tensor to list before jsonify
        filtered_labels = []
        filtered_outputs = []
        for index, output in enumerate(probabilities_3[0]):
            if labels_list[index] == "None":
                continue
            if output >= 0.25:
                filtered_labels.append(labels_list[index])
                filtered_outputs.append(output.item())  # Convert the tensor to a Python number

        # Store the filtered labels and outputs in the map with the corresponding sentence
        output_map[sentence] = {"Labels": filtered_labels, "Probabilities": filtered_outputs}

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
