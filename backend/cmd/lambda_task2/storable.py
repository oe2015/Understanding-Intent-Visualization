from abc import ABC, abstractmethod


class StorableDynamoDB(ABC):
    @abstractmethod
    def PK(self) -> str:
        """Primary Key for DynamoDB storage. Must be implemented by subclasses."""
        pass

    def SK(self) -> str:
        """
        Sort Key for DynamoDB storage. Optional to implement.
        Raises error if called but not implemented by subclass.
        """
        raise NotImplementedError("Sort Key (SK) method is optional and not implemented.")

    def GS1PK(self) -> str:
        """
        Global Secondary Index 1 Partition Key for DynamoDB. Optional to implement.
        Raises error if called but not implemented by subclass.
        """
        raise NotImplementedError(
            "GS1 Partition Key (GS1PK) method is optional and not implemented."
        )

    @abstractmethod
    def items(self) -> dict:
        """
        Convert the object properties to a dictionary format for DynamoDB storage.
        Must be implemented by subclasses to fit their specific attributes.
        """
        pass
