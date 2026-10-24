from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class SummarizeRequest(BaseModel):
    text: str = Field(..., description="The legislative bill or document text to summarize")
    title: Optional[str] = Field(None, description="Optional title of the bill or document")
    max_tokens: int = Field(512, ge=64, le=2048, description="Maximum tokens for the generated summary")
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="Sampling temperature")
    provider: str = Field("sovereign", description="Which engine to use: 'sovereign' or 'gemini'")


class SummarizeResponse(BaseModel):
    summary: str
    model: str
    provider: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


class QARequest(BaseModel):
    context: str = Field(..., description="Document or bill text context")
    question: str = Field(..., description="Question regarding the context")
    provider: str = Field("sovereign", description="Engine: 'sovereign' or 'gemini'")


class QAResponse(BaseModel):
    answer: str
    model: str
    provider: str
    latency_ms: float


class ArenaRequest(BaseModel):
    text: str = Field(..., description="Bill or text to test on both engines simultaneously")
    max_tokens: int = Field(512, ge=64, le=2048, default=512)
    temperature: float = Field(0.2, ge=0.0, le=1.0, default=0.2)


class ArenaModelResult(BaseModel):
    summary: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd_est: float


class ArenaResponse(BaseModel):
    sovereign: ArenaModelResult
    frontier: ArenaModelResult
    faster_model: str
    latency_delta_ms: float
    winner_summary: str


class BenchmarkStatsResponse(BaseModel):
    rouge_scores: Dict[str, Dict[str, float]]
    latency_percentiles: Dict[str, Dict[str, float]]
    cost_crossover: Dict[str, Any]


class SystemHealthResponse(BaseModel):
    status: str
    sovereign_backend: str
    phoenix_tracing: bool
    gemini_available: bool
