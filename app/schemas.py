from pydantic import BaseModel
from typing import Dict, List, Optional

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[int]
    groundedness: Optional[bool] = None
    grounded_top1: Optional[bool] = None
    faithfulness: Optional[bool] = None
    latency: Optional[Dict[str, float]] = None