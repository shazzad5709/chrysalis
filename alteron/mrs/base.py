from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMR(ABC):
    def __init__(self) -> None:
        self._last_skip_reason: str | None = None

    @property
    def last_skip_reason(self) -> str | None:
        return self._last_skip_reason

    def clear_skip_reason(self) -> None:
        self._last_skip_reason = None

    def set_skip_reason(self, reason: str) -> None:
        self._last_skip_reason = reason

    @property
    @abstractmethod
    def mr_id(self) -> str:
        """Return the unique MR identifier."""

    @property
    @abstractmethod
    def subtasks(self) -> list[str]:
        """Return the supported subtasks for this MR."""

    @abstractmethod
    def transform(self, source_input: str | dict, seed: int = 42) -> str | dict | None:
        """Transform an input or return None when the input should be skipped."""

    @abstractmethod
    def check_pass(self, source_output: dict, followup_output: dict) -> bool:
        """Evaluate whether the follow-up output satisfies the MR relation."""

    @abstractmethod
    def automated_checks(self, source_input: str | dict, followup_input: str | dict) -> bool:
        """Return True when the generated pair passes all automated checks."""
