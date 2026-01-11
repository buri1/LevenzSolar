from typing import Literal, Optional
from pydantic import BaseModel, Field

class ClassificationResult(BaseModel):
    product_id: str
    product_name: str
    is_pv_module: bool
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0", alias="Confidence")
    reasoning: str = Field(description="Short semantic reasoning", alias="Reasoning")
    
    # Power extraction fields for CO2 calculations
    power_watts: Optional[int] = Field(
        default=None, 
        alias="power_watts",
        description="Extracted power in Watts (e.g., 450 for 450Wp module or 14500 for 14.5kWp system)"
    )
    quantity: Optional[int] = Field(
        default=None,
        alias="quantity",
        description="Number of modules (from text or quantity column)"
    )
    total_power_watts: Optional[int] = Field(
        default=None,
        alias="total_power_watts", 
        description="Total power = power_watts * quantity"
    )
    power_source: Optional[str] = Field(
        default=None,
        alias="power_source",
        description="Source of power extraction: 'module_wp', 'anlage_kwp', 'productcode', or null"
    )

    model_config = {
        "populate_by_name": True
    }
