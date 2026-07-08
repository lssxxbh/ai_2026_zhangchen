from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class Summary(BaseModel):
    title: str
    diagnosis: str
    risk_level: str
    confidence: float
    reason: str
    health_score: int


class Indicator(BaseModel):
    name: str
    abbreviation: str
    value: float
    unit: str
    reference: str
    status: str


class Agent(BaseModel):
    department: str
    risk_level: str
    confidence: float
    summary: str
    recommendation: List[str]


class SimilarCase(BaseModel):
    case_id: str
    disease: str
    similarity: float


class KnowledgeGraphItem(BaseModel):
    disease: str
    relation: str
    entity: str
    value: str


class KnowledgeItem(BaseModel):
    source: str
    content: str


class Recommendation(BaseModel):
    diet: List[str]
    exercise: List[str]
    follow_up: List[str]


class Visualization(BaseModel):
    summary: Summary
    indicators: List[Indicator]
    agents: List[Agent]
    similar_cases: List[SimilarCase]
    knowledge_graph: List[KnowledgeGraphItem]
    knowledge: List[KnowledgeItem]
    recommendation: Recommendation
    warning: str


class ReportParseResponse(BaseModel):
    success: bool
    raw_text: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    visualization: Optional[Visualization] = None
