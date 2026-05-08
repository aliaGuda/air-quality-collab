from typing import Dict, List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of model input features.",
    )


class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    confidence: float
    latency_ms: float
    feature_hash: str
    prediction: float


class BatchPredictionRequest(BaseModel):
    records: List[PredictionRequest]