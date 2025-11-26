from pydantic import BaseModel, Field


class Detection(BaseModel):
    """
    Represents a single object detection result.

    Attributes:
        box (list[float]): Bounding box coordinates in the format [x_min, y_min, x_max, y_max] in pixels.
        confidence (float): Confidence score of the detection, ranging from 0 to 1.
        class_id (int): Identifier for the detected class.
        class_name (str): Name of the detected class.
    """

    box: list[float] = Field(description="[x_min, y_min, x_max, y_max] in pixels")
    confidence: float = Field(description="Confidence score of the detection")
    class_id: int
    class_name: str


class PredictionResponse(BaseModel):
    """
    Represents the response structure for a prediction request.

    Attributes:
        status (str): Status of the prediction request (e.g., "success").
        num_detections (int): Total number of detections in the response.
        detections (list[Detection]): List of detection results.
    """

    status: str = "success"
    num_detections: int
    detections: list[Detection]
