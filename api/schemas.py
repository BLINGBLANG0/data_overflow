"""
Pydantic schemas for the Insurance Bundle Recommender API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class PolicyInput(BaseModel):
    """Single policy holder input for prediction."""

    User_ID: str = Field(..., description="Unique user identifier")
    Policy_Start_Year: Optional[int] = Field(None, description="Year the policy starts")
    Policy_Start_Month: Optional[str] = Field(None, description="Month name (e.g. 'January')")
    Policy_Start_Week: Optional[int] = Field(None)
    Policy_Start_Day: Optional[int] = Field(None)
    Grace_Period_Extensions: Optional[int] = Field(None)
    Previous_Policy_Duration_Months: Optional[int] = Field(None)
    Adult_Dependents: Optional[int] = Field(None)
    Child_Dependents: Optional[int] = Field(None)
    Infant_Dependents: Optional[int] = Field(None)
    Region_Code: Optional[str] = Field(None)
    Existing_Policyholder: Optional[int] = Field(None)
    Previous_Claims_Filed: Optional[int] = Field(None)
    Years_Without_Claims: Optional[int] = Field(None)
    Policy_Amendments_Count: Optional[int] = Field(None)
    Underwriting_Processing_Days: Optional[int] = Field(None)
    Vehicles_on_Policy: Optional[int] = Field(None)
    Custom_Riders_Requested: Optional[int] = Field(None)
    Broker_Agency_Type: Optional[str] = Field(None)
    Deductible_Tier: Optional[str] = Field(None)
    Acquisition_Channel: Optional[str] = Field(None)
    Payment_Schedule: Optional[str] = Field(None)
    Employment_Status: Optional[str] = Field(None)
    Estimated_Annual_Income: Optional[float] = Field(None)
    Days_Since_Quote: Optional[int] = Field(None)
    Policy_Cancelled_Post_Purchase: Optional[int] = Field(None)
    Broker_ID: Optional[float] = Field(None)
    Employer_ID: Optional[float] = Field(None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "User_ID": "USR_12345",
                    "Policy_Start_Year": 2016,
                    "Policy_Start_Month": "March",
                    "Policy_Start_Week": 12,
                    "Policy_Start_Day": 15,
                    "Grace_Period_Extensions": 0,
                    "Previous_Policy_Duration_Months": 3,
                    "Adult_Dependents": 2,
                    "Child_Dependents": 1,
                    "Infant_Dependents": 0,
                    "Region_Code": "ABJ",
                    "Existing_Policyholder": 0,
                    "Previous_Claims_Filed": 0,
                    "Years_Without_Claims": 0,
                    "Policy_Amendments_Count": 1,
                    "Underwriting_Processing_Days": 3,
                    "Vehicles_on_Policy": 1,
                    "Custom_Riders_Requested": 0,
                    "Broker_Agency_Type": "National_Corporate",
                    "Deductible_Tier": "Tier_2_Moderate_Ded",
                    "Acquisition_Channel": "Direct_Website",
                    "Payment_Schedule": "Monthly_EFT",
                    "Employment_Status": "Employed_FullTime",
                    "Estimated_Annual_Income": 35000.0,
                    "Days_Since_Quote": 5,
                    "Policy_Cancelled_Post_Purchase": 0,
                    "Broker_ID": 42.0,
                    "Employer_ID": 100.0,
                }
            ]
        }
    }


class PredictionResult(BaseModel):
    """Single prediction result."""

    User_ID: str
    predicted_bundle: int = Field(
        ..., ge=0, le=9, description="Predicted coverage bundle (0-9)"
    )
    confidence: float = Field(..., description="Prediction confidence (max probability)")
    probabilities: dict[str, float] = Field(
        ..., description="Per-class probabilities"
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request (list of policies)."""

    policies: List[PolicyInput]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictionResult]
    count: int
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_size_mb: float
    num_features: int
    version: str


BUNDLE_DESCRIPTIONS = {
    0: "Basic Liability Only",
    1: "Standard Coverage",
    2: "Standard Plus",
    3: "Enhanced Coverage",
    4: "Comprehensive",
    5: "Premium Protection",
    6: "Family Shield",
    7: "Executive Suite",
    8: "Specialty Coverage",
    9: "Ultra Premium",
}
