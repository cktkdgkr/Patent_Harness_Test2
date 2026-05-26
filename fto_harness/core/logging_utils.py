"""
Processing log utilities for the FTO Harness.

Every agent action — success, failure, uncertainty — is recorded.
Failures and unknowns automatically trigger NEEDS_HUMAN_REVIEW escalation.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any

from .models import AgentName, ProcessingLogEntry


class ProcessingLogger:
    def __init__(self) -> None:
        self.entries: list[ProcessingLogEntry] = []
        self._verbose = True

    def log(
        self,
        agent: AgentName | str,
        stage: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
        **extra: Any,
    ) -> ProcessingLogEntry:
        agent_str = agent.value if isinstance(agent, AgentName) else agent
        merged = dict(details or {})
        merged.update(extra)
        entry = ProcessingLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            agent=agent_str,
            stage=stage,
            status=status,
            message=message,
            details=merged,
        )
        self.entries.append(entry)
        if self._verbose:
            marker = "OK" if status == "success" else "!!" if status == "failure" else "??"
            print(f"[{marker}] [{agent_str}] {stage}: {message}", file=sys.stderr)
        return entry

    def log_success(
        self, agent: AgentName | str, stage: str, message: str, **details: Any
    ) -> ProcessingLogEntry:
        return self.log(agent, stage, "success", message, details)

    def log_failure(
        self, agent: AgentName | str, stage: str, message: str, **details: Any
    ) -> ProcessingLogEntry:
        return self.log(agent, stage, "failure", message, details)

    def log_uncertain(
        self, agent: AgentName | str, stage: str, message: str, **details: Any
    ) -> ProcessingLogEntry:
        """Log an uncertain outcome — these always need human review."""
        return self.log(agent, stage, "uncertain", message, details)

    def log_skipped(
        self, agent: AgentName | str, stage: str, message: str, **details: Any
    ) -> ProcessingLogEntry:
        return self.log(agent, stage, "skipped", message, details)

    def get_failures(self) -> list[ProcessingLogEntry]:
        return [e for e in self.entries if e.status == "failure"]

    def get_uncertainties(self) -> list[ProcessingLogEntry]:
        return [e for e in self.entries if e.status == "uncertain"]

    def has_any_issues(self) -> bool:
        return any(e.status in ("failure", "uncertain") for e in self.entries)

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_list(), f, indent=2, ensure_ascii=False)
