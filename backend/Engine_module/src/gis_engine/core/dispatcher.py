from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class AnalysisDispatcher:
    """
    Determines which analysis branch is executed.

    The dispatcher does not implement scientific processing.
    It only routes the existing production pipeline according
    to the configured analysis mode.
    """

    analysis_mode: str

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AnalysisDispatcher":
        mode = str(config.get("analysis_mode", "wetland")).lower()

        if mode not in {"vegetation", "wetland"}:
            raise ValueError(
                f"Unsupported analysis mode: {mode!r}. "
                "Expected 'vegetation' or 'wetland'."
            )

        return cls(analysis_mode=mode)

    @property
    def is_vegetation(self) -> bool:
        return self.analysis_mode == "vegetation"

    @property
    def is_wetland(self) -> bool:
        return self.analysis_mode == "wetland"

    @property
    def requires_wetland_context(self) -> bool:
        return self.is_wetland
