"""Request/response schemas for the prediction endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# A realistic CICIDS2017 flow-feature example, purely for OpenAPI docs -- the
# authoritative, currently-required feature set is GET /api/v1/model.
_EXAMPLE_FEATURES = {
    "Destination Port": 443.0,
    "Flow Duration": 84852.0,
    "Bwd Packet Length Max": 1460.0,
    "Bwd Packet Length Mean": 487.33,
    "Bwd Packet Length Std": 601.34,
    "Flow IAT Std": 21023.5,
    "Flow IAT Max": 84763.0,
    "Fwd IAT Total": 84852.0,
    "Fwd IAT Std": 21023.5,
    "Fwd IAT Max": 84763.0,
    "Bwd IAT Total": 0.0,
    "Bwd IAT Std": 0.0,
    "Bwd IAT Max": 0.0,
    "Fwd PSH Flags": 0.0,
    "Max Packet Length": 1460.0,
    "Packet Length Mean": 487.33,
    "Packet Length Std": 601.34,
    "FIN Flag Count": 0.0,
    "SYN Flag Count": 1.0,
    "PSH Flag Count": 1.0,
    "ACK Flag Count": 1.0,
    "URG Flag Count": 0.0,
    "Average Packet Size": 585.0,
    "Avg Bwd Segment Size": 487.33,
    "Init_Win_bytes_forward": 8192.0,
    "Init_Win_bytes_backward": 65535.0,
    "min_seg_size_forward": 32.0,
    "Idle Mean": 0.0,
    "Idle Max": 0.0,
    "Idle Min": 0.0,
}

_EXAMPLE_RESPONSE = {
    "predicted_class": "BENIGN",
    "confidence": 0.98,
    "class_probabilities": {"BENIGN": 0.98, "DoS Hulk": 0.01, "PortScan": 0.01},
    "model_version": "v1",
}


class PredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"features": _EXAMPLE_FEATURES}]})

    features: dict[str, float] = Field(
        ...,
        description=(
            "CICIDS2017 flow feature name -> value. Must match the currently loaded "
            "model's feature set exactly -- see GET /api/v1/model for the authoritative list."
        ),
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(), json_schema_extra={"examples": [_EXAMPLE_RESPONSE]}
    )

    predicted_class: str = Field(..., description="Predicted label, e.g. 'BENIGN' or 'DoS Hulk'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of the predicted class")
    class_probabilities: dict[str, float] | None = Field(
        None, description="Probability of every class, when the model supports it"
    )
    model_version: str = Field(
        ..., description="Version of the model that produced this prediction"
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"instances": [_EXAMPLE_FEATURES]}]})

    instances: list[dict[str, float]] = Field(
        ...,
        min_length=1,
        description="List of feature dicts, one per row -- same schema as POST /predict.",
    )


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"predictions": [_EXAMPLE_RESPONSE], "count": 1}]}
    )

    predictions: list[PredictionResponse]
    count: int = Field(..., description="Number of predictions returned")
