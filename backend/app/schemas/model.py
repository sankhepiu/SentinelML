from pydantic import BaseModel, ConfigDict, Field

_EXAMPLE = {
    "model_type": "lightgbm",
    "model_version": "v1",
    "feature_names": ["Destination Port", "Flow Duration", "..."],
    "label_classes": ["BENIGN", "DoS Hulk", "DoS GoldenEye", "..."],
    "metrics": {"accuracy": 0.9997, "f1_macro": 0.9976, "roc_auc_ovr_macro": 0.99999},
}


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), json_schema_extra={"examples": [_EXAMPLE]})

    model_type: str = Field(..., description="Winning algorithm from training, e.g. 'lightgbm'")
    model_version: str = Field(..., description="Model registry version, e.g. 'v1'")
    feature_names: list[str] = Field(
        ..., description="Exact feature names and order the model expects"
    )
    label_classes: list[str] = Field(..., description="Every class name the model can predict")
    metrics: dict[str, float] = Field(..., description="Held-out test-set metrics from training")
