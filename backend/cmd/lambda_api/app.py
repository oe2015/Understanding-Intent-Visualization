import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime

import aws_lambda_wsgi
import boto3
from aws_lambda_powertools.logging import Logger
from dynamodbinterface import DynamoDBInterface, StorableDynamoDB
from flask import Flask, jsonify, request


@dataclass
class Entry(StorableDynamoDB):
    id: int
    task: str
    status: str
    payload: dict
    last_updated: str

    def items(self):
        """Returns the items of the dictionary representing the data class."""
        return asdict(self).items()

    def PK(self):
        return f"id#{self.id}"


# Initialize logging
logger = Logger()

# Initialize the Flask app
app = Flask(__name__)
logger.info("Flask app initialized")


# Health Check Handler
@app.route("/health", methods=["GET"])
def index():
    logger.info("GET /health")
    return jsonify({"message": "FRAPPE API is running"})


# Task 1 Handler
@app.route("/run/task1", methods=["POST"])
def task1_run():
    logger.info("POST /run/task1")

    # Request will serve as Payload
    payload = request.json
    logger.info(f"Received payload: {payload}")

    # Convert the payload to string
    payload_str = json.dumps(payload)

    # Assign an ID to the task
    task_id = str(uuid.uuid4()).replace("-", "")

    # Create an entry in the database
    ddbi = DynamoDBInterface(
        dynamodb_svc=boto3.client("dynamodb", region_name="us-east-1"),
        table="frappe-db-devhasaniqbal-us-east-1",
    )
    ddbi.create_or_update(
        Entry(
            id=task_id,
            task="task1",
            status="PENDING",
            payload=payload_str,
            last_updated=str(datetime.now()),
        )
    )

    # Trigger the task
    boto3.client("lambda", region_name="us-east-1").invoke(
        FunctionName="arn:aws:lambda:us-east-1:904233118707:function:frappe-task1-devhasaniqbal-us-east-1",
        InvocationType="Event",
        Payload='{"task_id": "%s"}' % task_id,
    )

    return jsonify({"task_id": task_id})


# Task 2 Handler
@app.route("/run/task2", methods=["POST"])
def task2_run():
    logger.info("POST /run/task2")

    # Request will serve as Payload
    payload = request.json
    logger.info(f"Received payload: {payload}")

    # Convert the payload to string
    payload_str = json.dumps(payload)

    # Assign an ID to the task
    task_id = str(uuid.uuid4()).replace("-", "")

    # Create an entry in the database
    ddbi = DynamoDBInterface(
        dynamodb_svc=boto3.client("dynamodb", region_name="us-east-1"),
        table="frappe-db-devhasaniqbal-us-east-1",
    )
    ddbi.create_or_update(
        Entry(
            id=task_id,
            task="task2",
            status="PENDING",
            payload=payload_str,
            last_updated=str(datetime.now()),
        )
    )

    # Trigger the task
    boto3.client("lambda", region_name="us-east-1").invoke(
        FunctionName="arn:aws:lambda:us-east-1:904233118707:function:frappe-task2-devhasaniqbal-us-east-1",
        InvocationType="Event",
        Payload='{"task_id": "%s"}' % task_id,
    )

    return jsonify({"task_id": task_id})


# Task 3 Handler
@app.route("/run/task3", methods=["POST"])
def task3_run():
    logger.info("POST /run/task3")

    # Request will serve as Payload
    payload = request.json
    logger.info(f"Received payload: {payload}")

    # Convert the payload to string
    payload_str = json.dumps(payload)

    # Assign an ID to the task
    task_id = str(uuid.uuid4()).replace("-", "")

    # Create an entry in the database
    ddbi = DynamoDBInterface(
        dynamodb_svc=boto3.client("dynamodb", region_name="us-east-1"),
        table="frappe-db-devhasaniqbal-us-east-1",
    )
    ddbi.create_or_update(
        Entry(
            id=task_id,
            task="task3",
            status="PENDING",
            payload=payload_str,
            last_updated=str(datetime.now()),
        )
    )

    # Trigger the task
    boto3.client("lambda", region_name="us-east-1").invoke(
        FunctionName="arn:aws:lambda:us-east-1:904233118707:function:frappe-task3-devhasaniqbal-us-east-1",
        InvocationType="Event",
        Payload='{"task_id": "%s"}' % task_id,
    )

    return jsonify({"task_id": task_id})


# Get Run Handler
@app.route("/run/<task_id>", methods=["GET"])
def get_run(task_id):
    logger.info(f"GET /run/{task_id}")

    # Create an entry in the database
    ddbi = DynamoDBInterface(
        dynamodb_svc=boto3.client("dynamodb", region_name="us-east-1"),
        table="frappe-db-devhasaniqbal-us-east-1",
    )
    response = ddbi.get(f"id#{task_id}")

    if response is None:
        return jsonify({"error": "task not found"}), 404

    return jsonify(response)


# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "internal server error"}), 500


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


if __name__ == "__main__":
    app.run()
