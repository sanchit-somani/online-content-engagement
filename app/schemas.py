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
    will_get_answered: bool
    probability_answered: float
    threshold: float
    top_drivers: List[str]
