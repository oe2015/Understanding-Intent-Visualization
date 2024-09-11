from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from storable import StorableDynamoDB


class DynamoDBInterface:
    def __init__(self, dynamodb_svc: BaseClient, table: str):
        self.table: str = table
        self.dynamodb_svc: BaseClient = dynamodb_svc

    def _serialize_attributes_to_map(self, storable: StorableDynamoDB):
        # Serialize the basic attributes
        serializer = TypeSerializer()
        av = {k: serializer.serialize(v) for k, v in storable.items()}

        # Always add the primary key
        av["PK"] = {"S": storable.PK()}

        # Optionally add the sort key, if implemented
        if hasattr(storable, "SK") and callable(getattr(storable, "SK")):
            try:
                sk_value = storable.SK()
                if sk_value is not None:
                    av["SK"] = {"S": sk_value}
            except NotImplementedError:
                # If SK is not implemented, we do nothing
                pass

        # Optionally add the GS1 partition key, if implemented
        if hasattr(storable, "GS1PK") and callable(getattr(storable, "GS1PK")):
            try:
                gs1pk_value = storable.GS1PK()
                if gs1pk_value is not None:
                    av["GS1PK"] = {"S": gs1pk_value}
            except NotImplementedError:
                # If GS1PK is not implemented, we do nothing
                pass

        return av

    def _deserialize_map_to_attributes(self, map):
        deserializer = TypeDeserializer()
        attributes = {k: deserializer.deserialize(v) for k, v in map.items()}
        return attributes

    def create_or_update(self, storable: StorableDynamoDB):
        try:
            _ = self.dynamodb_svc.put_item(
                TableName=self.table, Item=self._serialize_attributes_to_map(storable)
            )
            return
        except ClientError as e:
            raise Exception(f"DynamoDB.Interface.create_or_update failed: {e}") from e

    def get(self, pk: str):
        try:
            response = self.dynamodb_svc.get_item(TableName=self.table, Key={"PK": {"S": pk}})
            if "Item" not in response:
                return None

            return self._deserialize_map_to_attributes(response["Item"])
        except ClientError as e:
            raise Exception(f"DynamoDB.Interface.get failed: {e}") from e
