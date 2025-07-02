from pydantic import BaseModel
from typing import List, Optional

class PredictionInput(BaseModel):
    AccountId: str
    Amount: float
    Value: float
    ProductCategory: str
    ChannelId: str
    CountryCode: str
    TransactionStartTime: str
    
class PredictionOutput(BaseModel):
    customer_id: str
    risk_probability: float
    risk_category: str
    model_version: str