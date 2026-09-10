from abc import ABC, abstractmethod
from typing import Any


class AnalysisAdapter(ABC):
	name: str

	@abstractmethod
	def execute(self, *, image_path: str, question: str | None) -> dict[str, Any]:
		"""Run an integration when the corresponding component is available."""
		raise NotImplementedError


class AgentAdapter(AnalysisAdapter):
	name = "person1_agent"

	def execute(self, *, image_path: str, question: str | None) -> dict[str, Any]:
		return {
			"output": "Example agent result",
			"notice": "Development placeholder; not a real satellite-AI result.",
		}


class VisionAdapter(AnalysisAdapter):
	name = "person3_vision"

	def execute(self, *, image_path: str, question: str | None) -> dict[str, Any]:
		return {
			"output": "Example vision result",
			"notice": "Development placeholder; not a real satellite-AI result.",
		}


class ChangeDetectionAdapter(AnalysisAdapter):
	name = "person4_change_detection"

	def execute(self, *, image_path: str, question: str | None) -> dict[str, Any]:
		return {
			"output": "Example change-detection result",
			"notice": "Development placeholder; not a real satellite-AI result.",
		}


ANALYSIS_ADAPTERS: tuple[AnalysisAdapter, ...] = (
	AgentAdapter(),
	VisionAdapter(),
	ChangeDetectionAdapter(),
)