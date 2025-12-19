from typing import Literal
from pydantic import BaseModel, Field

class ClassificationResult(BaseModel):
    product_id: str
    product_name: str
    is_pv_module: bool
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0", alias="Confidence")
    reasoning: str = Field(description="Short semantic reasoning", alias="Reasoning")

    model_config = {
        "populate_by_name": True
    }
