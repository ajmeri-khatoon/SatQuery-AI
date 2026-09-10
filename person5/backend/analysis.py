from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..integration.adapters import ANALYSIS_ADAPTERS
from ..models import Analysis, Execution, Image, Result, User
from ..schemas.analysis import (
	AnalysisCreateResponse,
	AnalysisResultResponse,
	AnalyzeRequest,
	ExecutionResponse,
	QueryRequest,
)
from ..services.analysis_execution import execute_analysis
from .auth import get_current_user
from .database import get_db


router = APIRouter(tags=["analysis"])


def _get_user_image(image_id: int, user_id: int, db: Session) -> Image:
	image = db.scalar(select(Image).where(Image.id == image_id, Image.user_id == user_id))
	if image is None:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
	return image


def _create_analysis(
	*, image: Image, user: User, question: str | None, db: Session
) -> AnalysisCreateResponse:
	analysis = Analysis(user_id=user.id, image_id=image.id, question=question, status="pending")
	db.add(analysis)
	db.flush()

	for _adapter in ANALYSIS_ADAPTERS:
		db.add(Execution(analysis_id=analysis.id, status="pending"))

	db.commit()
	db.refresh(analysis)
	return AnalysisCreateResponse(analysis_id=analysis.id, status=analysis.status)


@router.post("/query", response_model=AnalysisCreateResponse, status_code=status.HTTP_201_CREATED)
def create_query(
	request: QueryRequest,
	current_user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
) -> AnalysisCreateResponse:
	image = _get_user_image(request.image_id, current_user.id, db)
	return _create_analysis(image=image, user=current_user, question=request.question, db=db)


@router.post("/analyze", response_model=AnalysisCreateResponse, status_code=status.HTTP_201_CREATED)
def create_analysis(
	request: AnalyzeRequest,
	current_user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
) -> AnalysisCreateResponse:
	image = _get_user_image(request.image_id, current_user.id, db)
	return _create_analysis(image=image, user=current_user, question=request.question, db=db)


def _get_user_analysis(analysis_id: int, user_id: int, db: Session) -> Analysis:
	analysis = db.scalar(
		select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user_id)
	)
	if analysis is None:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
	return analysis


@router.get("/result/{analysis_id}", response_model=AnalysisResultResponse)
def get_result(
	analysis_id: int,
	current_user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
) -> AnalysisResultResponse:
	analysis = _get_user_analysis(analysis_id, current_user.id, db)
	results = db.scalars(
		select(Result).where(Result.analysis_id == analysis.id).order_by(Result.created_at)
	).all()
	return AnalysisResultResponse(
		analysis_id=analysis.id,
		image_id=analysis.image_id,
		question=analysis.question,
		status=analysis.status,
		results=results,
	)


@router.post("/analyze/{analysis_id}/run", response_model=AnalysisCreateResponse)
def run_analysis(
	analysis_id: int,
	current_user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
) -> AnalysisCreateResponse:
	analysis = _get_user_analysis(analysis_id, current_user.id, db)
	if analysis.status != "pending":
		raise HTTPException(
			status_code=status.HTTP_409_CONFLICT,
			detail=f"Analysis is already {analysis.status}",
		)
	if analysis.image_id is None:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Analysis has no image")

	image = _get_user_image(analysis.image_id, current_user.id, db)
	analysis = execute_analysis(analysis, image, db)
	return AnalysisCreateResponse(analysis_id=analysis.id, status=analysis.status)


@router.get("/execution/{analysis_id}", response_model=list[ExecutionResponse])
def get_execution(
	analysis_id: int,
	current_user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
) -> list[ExecutionResponse]:
	analysis = _get_user_analysis(analysis_id, current_user.id, db)
	executions = db.scalars(
		select(Execution)
		.where(Execution.analysis_id == analysis.id)
		.order_by(Execution.id)
	).all()
	return [
		ExecutionResponse(
			id=execution.id,
			step=ANALYSIS_ADAPTERS[index].name if index < len(ANALYSIS_ADAPTERS) else "unknown",
			status=execution.status,
			started_at=execution.started_at,
			completed_at=execution.completed_at,
			error_message=execution.error_message,
			created_at=execution.created_at,
		)
		for index, execution in enumerate(executions)
	]