from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..integration.adapters import ANALYSIS_ADAPTERS
from ..models import Analysis, Execution, Image, Result


def _now() -> datetime:
	return datetime.now(timezone.utc)


def execute_analysis(analysis: Analysis, image: Image, db: Session) -> Analysis:
	"""Execute the configured development adapters and persist an audit trail."""
	analysis.status = "running"
	db.commit()

	executions = db.scalars(
		select(Execution)
		.where(Execution.analysis_id == analysis.id)
		.order_by(Execution.id)
	).all()

	try:
		for index, adapter in enumerate(ANALYSIS_ADAPTERS):
			execution = executions[index]
			execution.status = "running"
			execution.started_at = _now()
			db.commit()

			output = adapter.execute(image_path=image.file_path, question=analysis.question)
			db.add(Result(analysis_id=analysis.id, result_type=adapter.name, data=output))
			execution.status = "completed"
			execution.completed_at = _now()
			db.commit()

		analysis.status = "completed"
		db.commit()
	except Exception as error:
		db.rollback()
		failed_execution = db.scalar(
			select(Execution)
			.where(Execution.analysis_id == analysis.id, Execution.status == "running")
			.order_by(Execution.id.desc())
		)
		if failed_execution is not None:
			failed_execution.status = "failed"
			failed_execution.completed_at = _now()
			failed_execution.error_message = str(error)
		analysis.status = "failed"
		db.commit()

	return analysis