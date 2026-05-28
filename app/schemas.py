from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[int]
    groundedness: Optional[bool] = None
    grounded_top1: Optional[bool] = None
    faithfulness: Optional[bool] = None
    llm_groundedness: Optional[bool] = None