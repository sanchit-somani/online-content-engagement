from pydantic import BaseModel
from typing import List, Optional

class PredictRequest(BaseModel):
    title: str
    body: str
    tags: List[str] = []
    # Optional: allow client to provide time, else you can default
    hour: Optional[int] = 12
    weekday: Optional[int] = 2  # 0=Mon ... 6=Sun

class PredictResponse(BaseModel):
    valid_input: bool
    validation_reasons: List[str]

    will_get_answered: bool
    probability_answered: float
    adjusted_probability_answered: float
    quality_score: float

    threshold: float
    top_drivers: List[str]
