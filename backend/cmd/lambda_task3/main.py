import json
from dataclasses import asdict, dataclass
from datetime import datetime

import boto3
import nltk
import torch
from aws_lambda_powertools.logging import Logger
from dynamodbinterface import DynamoDBInterface, StorableDynamoDB
from transformers import AutoTokenizer
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

# Define configuration constants
PROB_THRESHOLD = 0.1  # Adjust as needed
MODEL_PATH = "./models/model.pt"
TOKENIZER_PATH = "./models/xlm-roberta-large"
MAX_LENGTH_CONTENT = 512  # Adjust based on model's training


# Download NLTK data outside the handler
try:
    nltk.download("punkt_tab", download_dir="/tmp/nltk_data/", quiet=True)
    nltk.data.path.append("/tmp/nltk_data/")
    logger.info("NLTK punkt tokenizer downloaded")
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")
    raise e

# Load the tokenizer outside the handler
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    logger.info("Tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    raise e

# Load the TorchScript model outside the handler
try:
    persuasion_model = torch.jit.load(MODEL_PATH, map_location="cpu")
    persuasion_model.eval()
    logger.info("TorchScript model loaded and set to evaluation mode")
except Exception as e:
    logger.error(f"Failed to load TorchScript model: {e}")
    raise e


def chunk_text(tokenizer, text, chunk_size=512, overlap=50):
    """Utility to split the text into chunks of 512 tokens with overlap."""
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
        text = payload.get("text")
        if text is None:
            raise Exception("Text not found in the payload")

        # Initialize output map
        output_map = {}

        # Extract sentences from text
        sentences = extract_title_and_sentences(text)
        logger.info(f"Extracted sentences: {sentences}")
        
        for sentence in sentences:
            # Chunk the sentence using sliding window approach
            sentence_chunks = chunk_text(tokenizer, sentence)

            # Initialize list to store all chunk outputs
            all_chunk_outputs = []
            all_chunk_labels = []

            for chunk in sentence_chunks:
                # Tokenize the chunk
                inputs = tokenizer.encode_plus(
                    chunk,
                    add_special_tokens=True,
                    max_length=MAX_LENGTH_CONTENT,
                    padding="longest",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                inputs_3 = {
                    "content_input_ids": inputs["input_ids"],
                    "content_attention_mask": inputs["attention_mask"],
                }

                # Log input shapes and types
                for name, tensor in inputs_3.items():
                    logger.info(f"Input {name} shape: {tensor.shape}, dtype: {tensor.dtype}")

                # Pass the tokenized chunk to the model
                try:
                    outputs_3 = persuasion_model(**inputs_3)
                    logger.info(f"Raw model outputs for chunk: {outputs_3}")
                    probabilities_3 = torch.sigmoid(outputs_3)
                    logger.info(f"Probabilities after sigmoid: {probabilities_3}")
                except Exception as e:
                    logger.error(f"Error during model inference: {e}")
                    continue  # Skip to the next chunk

                # Check output dimensions
                if outputs_3.shape[1] != len(labels_list):
                    logger.error(
                        f"Output dimension {outputs_3.shape[1]} does not match number of labels {len(labels_list)}"
                    )
                    continue  # Skip or handle accordingly

                # Process the output and append to the lists for chunk outputs
                filtered_labels = []
                filtered_outputs = []
                for index, output in enumerate(probabilities_3[0]):
                    if labels_list[index] == "None":
                        continue
                    logger.info(f"Label: {labels_list[index]}, Probability: {output.item()}")
                    if output >= PROB_THRESHOLD:
                        filtered_labels.append(labels_list[index])
                        filtered_outputs.append(output.item())  # Convert tensor to Python float

                # Append chunk results
                all_chunk_labels.append(filtered_labels)
                all_chunk_outputs.append(filtered_outputs)

            # Concatenate chunk results into a single entry for the sentence
            concatenated_labels = [label for chunk_labels in all_chunk_labels for label in chunk_labels]
            concatenated_outputs = [output for chunk_outputs in all_chunk_outputs for output in chunk_outputs]

            # Store the concatenated labels and outputs in the map with the corresponding sentence
            output_map[sentence] = {
                "Labels": concatenated_labels, 
                "Probabilities": concatenated_outputs
            }

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
        "task_id": "3bfe734c82bc49cfaeeadc23a8b65218",
    }
    context = {}
    lambda_handler(event, context)
