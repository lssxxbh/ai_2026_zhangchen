# schemas/report.py
# -*- coding: utf-8 -*-

from pydantic import BaseModel
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
    abbreviation: str = ""
    value: float = 0.0
    unit: str = ""
    reference: str = "N/A"
    status: str = "N/A"


class Agent(BaseModel):
    department: str
    risk_level: str
    confidence: float
    summary: str
    recommendation: List[str] = []


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
    diet: List[str] = []
    exercise: List[str] = []
    follow_up: List[str] = []


class DepartmentSuggestion(BaseModel):
    name: str
    reason: str


class RecommendedTest(BaseModel):
    name: str
    reason: str
    priority: str = "建议"


class FollowUpPlan(BaseModel):
    time: str
    action: str


class Visualization(BaseModel):
    summary: Summary
    indicators: List[Indicator] = []
    agents: List[Agent] = []

    # 暂时保留字段，但 A 工程会主动清空，避免无关病例误导
    similar_cases: List[SimilarCase] = []

    knowledge_graph: List[KnowledgeGraphItem] = []
    knowledge: List[KnowledgeItem] = []
    recommendation: Recommendation
    warning: str

    # 新增：真正有用的展示字段
    departments: List[DepartmentSuggestion] = []
    recommended_tests: List[RecommendedTest] = []
    follow_up_plan: List[FollowUpPlan] = []
    red_flags: List[str] = []


class ReportParseResponse(BaseModel):
    success: bool
    raw_text: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    visualization: Optional[Visualization] = None
