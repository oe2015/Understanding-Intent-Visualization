import os

# Set Transformers cache directory
os.environ["HF_HOME"] = "/tmp/cache/"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/cache/"

import aws_lambda_wsgi
from flask import Flask, jsonify, request
from aws_lambda_powertools.logging import Logger

import nltk
import torch
from transformers import AutoConfig, AutoTokenizer
from models import CustomModel

# Initialize logging
logger = Logger()

# Initialize the Flask app
app = Flask(__name__)
logger.info("Flask app initialized")

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
logger.info("Labels list defined")


def load_models():
    # Download the NLTK punkt tokenizer
    nltk.download("punkt_tab", download_dir="/tmp/nltk_data/")
    logger.info("NLTK punkt tokenizer downloaded")


# Health Check Handler
@app.route("/health", methods=["GET"])
def index():
    logger.info("GET /health")
    return jsonify({"message": "FRAPPE API is running"})


# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "internal server error"}), 500


@app.route("/task1", methods=["POST"])
def task1():
    # Load the models
    nltk.download("punkt_tab", download_dir="/tmp/nltk_data/")

    # Load the models and tokenizers
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", cache_dir="/tmp/cache/")
    logger.info("Tokenizers loaded")

    # Load the model
    xlmrobertaconfig = AutoConfig.from_pretrained("xlm-roberta-base")
    custommodel = CustomModel(xlmrobertaconfig)
    logger.info("Model loaded")

    # Download state dict from s3 URL
    model_url = "https://frappe-devhasaniqbal-us-east-1.s3.us-east-1.amazonaws.com/best_model-v1.ckpt?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIGG%2B23ualrEiumzz2LgzPHOJ%2BowA%2F2X2H2%2F3IMpeL4inAiEAs8BldEEcCbZ%2BZoWi%2BMITbycQHMfpElKhpfxj18YnNhgqigMI4f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw5MDQyMzMxMTg3MDciDMfk5j%2FJ33VVo1MDwyreAiSlQAlM1%2FY7YXMqkcG4FRm3PdzElyxl6J0IYa83J5IXH8Ueq6vLCBAjzL2g9HMgXtSsP4eAaJDT0VWPOy7EIJDAjD1xPMahKu5TEYPedxhwarJV6r6yeZUNDOXObxHyRjxsmjgc%2F33HN69KAhDgtpCs2l9ie7KFg2ZHLoF6l%2Bpb9%2BdPXXDXC8s1SDppCblKuSao24Go%2B%2B%2BfxVXOdgTxyS8MzZhmGNzRrVpT7BOFbzqAzRwTwJ6w1l%2FAAYMAUVkPo2r7cPwTtH8ENH61woAz4IUs7HhjtYloA605hfDx7dmX1P1TWJC6o3WR7HFLNWecrOZoVe%2Br%2Bdaj2d1UdoDJfZJ7iMkGIrk%2F6wSgSW5qOBxe4a3f3b%2Frkjjt2N2xYe0PMQ10JDSucmgnMnC2%2Fw9aqC1touCjQUd7Q8y5jMtBzPyzjKoY1QlNyaPLsmWOAjcY1Mq5eHoaCMUzSVYKma8AMNLD47YGOrMCKT2AaKPOsBAeLcQXl74I6rcjRnMU9wMcLdIhsVqWLY4wOxLRtNSU2ZHenBacs9G%2F%2FFyWGD9%2FwPiJkZ%2BEb0na0dtc5ex%2BOP9h5A2lhZFG587loNH6Tg00J1lmDTgSz3Nl6eZxy%2FDPl2ByqhIlnwJwrPhv8gHkew59CRQ9%2Bqhkk7xLrMeXH7sBTUOQ5LZ4N%2FlulKBzd7H53yQjSoP3%2Fm6OQ0nA9F%2Fs3DocuwbgtHNYQdBPpWsbmGDeoRR1CZe%2Bd5L%2FNl1GCkt1nafwz3ikaC8FzNFOH4ziKdpnLrV36Rco4zBe%2BY7L9NsGeH57BC7g2pnNQ2D%2F8lVXZFSFFgo6lUGezD%2FDL0HE2YThakvEUro2kJneZxYlJ1qTxfi7O496TaLreOTj%2B8zA5XsDpnZhcpXUlJvF2A%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240904T234411Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIA5FCD6NPZX4MNUSR2%2F20240904%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=233a35a0f322ec351f609b0459200b6cac009ba001383d48510b324a22fe3ac5"
    model_state_dict = torch.hub.load_state_dict_from_url(
        model_url, map_location=torch.device("cpu"), model_dir="/tmp/cache/"
    )["state_dict"]
    logger.info("Model state dict loaded")

    # # Load the model state dict
    # model_state_dict = torch.load("./models/best_model-v1.ckpt", map_location=torch.device("cpu"))[
    #     "state_dict"
    # ]
    model_state_dict = {
        k.replace("model.model.", "model."): v for k, v in model_state_dict.items()
    }
    model_state_dict = {k.replace("model.fc.", "fc."): v for k, v in model_state_dict.items()}
    model_state_dict = {
        k: v for k, v in model_state_dict.items() if k != "model.embeddings.position_ids"
    }
    custommodel.load_state_dict(model_state_dict)
    logger.info("Model state dict loaded")

    # Set the model to evaluation mode
    custommodel.eval()
    logger.info("Model set to evaluation mode")

    # Get the text from the request
    text = request.json["text"]

    # Tokenize the text
    inputs = tokenizer(
        text,
        max_length=512,
        padding="longest",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Get model's prediction
    outputs = custommodel(inputs)
    predicted_probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()

    # Convert the prediction to JSON and return it
    return jsonify(predicted_probabilities)


def lambda_handler(event, context):
    # Convert API Gateway event to WSGI-like environ
    event["httpMethod"] = event["requestContext"]["http"]["method"]
    event["path"] = event["requestContext"]["http"]["path"]
    event["queryStringParameters"] = event.get("queryStringParameters", {})

    # Log the event
    logger.info(f"Received event: {event}")
    logger.info(f"Received context: {context}")

    # Pass the WSGI-like environ to the Flask app
    return aws_lambda_wsgi.response(app, event, context)
