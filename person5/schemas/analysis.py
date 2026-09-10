from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
	image_id: int
	question: str = Field(min_length=1)


class AnalyzeRequest(BaseModel):
	image_id: int
	question: str | None = Field(default=None, min_length=1)


class AnalysisCreateResponse(BaseModel):
	analysis_id: int
	status: str


class ResultItemResponse(BaseModel):
	model_config = ConfigDict(from_attributes=True)

	id: int
	result_type: str | None
	data: dict[str, Any] | None
	created_at: datetime


class AnalysisResultResponse(BaseModel):
	analysis_id: int
	image_id: int | None
	question: str | None
	status: str
	results: list[ResultItemResponse]


class ExecutionResponse(BaseModel):
	id: int
	step: str
	status: str
	started_at: datetime | None
	completed_at: datetime | None
	error_message: str | None
	created_at: datetime