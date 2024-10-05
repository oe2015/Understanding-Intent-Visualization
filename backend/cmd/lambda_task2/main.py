import json
from dataclasses import asdict, dataclass
from datetime import datetime

import boto3
import onnxruntime as ort
import numpy as np
from aws_lambda_powertools.logging import Logger
from dynamodbinterface import DynamoDBInterface, StorableDynamoDB
from transformers import AutoTokenizer, AutoConfig


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

# Define consistent max_length for title and content
TITLE_MAX_LENGTH = 128  # Replace with the model's expected length
CONTENT_MAX_LENGTH = 128  # Replace with the model's expected length

# Initialize ONNX Runtime session and tokenizer outside the handler for better performance
onnx_model_path = "./models/model.onnx"
tokenizer = AutoTokenizer.from_pretrained("./models/xlm-roberta-base")

try:
    session = ort.InferenceSession(onnx_model_path)
    logger.info("ONNX model loaded successfully")
    # Log model input details
    for input in session.get_inputs():
        logger.info(
            f"Model expects input - Name: {input.name}, Shape: {input.shape}, Type: {input.type}"
        )
except Exception as e:
    logger.error(f"Failed to load ONNX model: {e}")
    raise e

def chunk_text(tokenizer, text, chunk_size=512, overlap=50):
    """Utility to split the content into chunks of 512 tokens with overlap."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk, clean_up_tokenization_spaces=True))
    return chunks

def lambda_handler(event, context):
    # Log the event
    logger.info(f"Received event: {event}")
    logger.info(f"Received context: {context}")

    try:
        # Get the task_id from the event
        task_id = event["task_id"]

        # Initialize DynamoDB interface
        ddbi = DynamoDBInterface(
            dynamodb_svc=boto3.client("dynamodb", region_name="us-east-1"),
            table="frappe-db-devhasaniqbal-us-east-1",
        )

        # Retrieve the entry from DynamoDB
        response = ddbi.get(f"id#{task_id}")

        # Get the payload from the response
        payload_str = response.get("payload")
        if payload_str is None:
            raise Exception("Payload not found in the database")

        # Parse the payload
        payload = json.loads(payload_str)
        title = payload.get("title")
        content = payload.get("content")
        if title is None or content is None:
            raise Exception("Title or content not found in the payload")

        # Tokenize the title
        inputs_title = tokenizer(
            title,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=TITLE_MAX_LENGTH,  # Ensure this matches the model's expectation
        )

        # Chunk the content
        content_chunks = chunk_text(tokenizer, content)

        # Prepare lists to store the concatenated results
        all_chunk_logits = []
        all_chunk_probabilities = []

        # Process each chunk of content and pass it through the model
        for chunk in content_chunks:
            # Tokenize each chunk
            inputs_chunk = tokenizer(
                chunk,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=CONTENT_MAX_LENGTH,  # Ensure this matches the model's expectation
            )

            # Prepare the input dictionary for ONNX Runtime
            ort_inputs = {
                "title_input_ids": inputs_title["input_ids"].astype(np.int64),
                "title_attention_mask": inputs_title["attention_mask"].astype(np.int64),
                "content_input_ids": inputs_chunk["input_ids"].astype(np.int64),
                "content_attention_mask": inputs_chunk["attention_mask"].astype(np.int64),
            }

            # Log input shapes and types
            for name, tensor in ort_inputs.items():
                logger.info(f"Input {name} shape: {tensor.shape}, dtype: {tensor.dtype}")

            # Perform inference for the chunk
            logger.info("Running inference with ONNX model")
            ort_outputs = session.run(None, ort_inputs)

            # Assuming the model outputs logits; apply sigmoid to get probabilities
            logits = ort_outputs[0]
            probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid function
            probabilities = probabilities.flatten().tolist()

            # Store logits and probabilities for later concatenation
            all_chunk_logits.append(logits)
            all_chunk_probabilities.extend(probabilities)  # Concatenate probabilities for all chunks

        # Filter the labels and outputs based on the threshold
        output_map = {}
        filtered_labels = []
        filtered_outputs = []
        for index, output in enumerate(all_chunk_probabilities):
            if index >= len(labels_list):
                logger.warning(f"Output index {index} exceeds labels list length")
                continue
            label = labels_list[index]
            if label == "None":
                continue
            if output >= 0.35:
                filtered_labels.append(label)
                filtered_outputs.append(output)
        logger.info(f"Filtered labels: {filtered_labels}")

        # Store the filtered labels and outputs in the map
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

    except Exception as e:
        logger.error(f"Error processing the request: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


if __name__ == "__main__":
    event = {
        "task_id": "0d602c3ac4e041dea5911fac36e13d22",
    }
    context = {}
    lambda_handler(event, context)
