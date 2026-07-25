from pydantic import BaseModel, Field


class ReadinessResponse(BaseModel):
    ready: bool = Field(
        ..., description="True once the model and preprocessing pipeline are loaded"
    )
    detail: str | None = Field(
        None, description="Why the service isn't ready, when `ready` is false"
    )
