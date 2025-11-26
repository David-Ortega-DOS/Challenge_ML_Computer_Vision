from pydantic import BaseModel, Field

class Detection(BaseModel):
    box: list[float] = Field(description="[x_min, y_min, x_max, y_max] en píxeles")
    confidence: float = Field(description="Confianza de la detección")
    class_id: int
    class_name: str

class PredictionResponse(BaseModel):
    status: str = "success"
    num_detections: int
    detections: list[Detection]