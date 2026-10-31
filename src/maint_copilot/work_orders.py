"""Work-order drafts that require explicit human approval before they take effect.

The agent can only *propose* a draft. A draft becomes a work order only through `approve`, called
by a person (never exposed as an agent tool), which records who approved it and when. Drafts must
cite pages the agent read, and each step must carry at least one citation.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Status = Literal["pending_approval", "approved", "rejected"]


class Step(BaseModel):
    instruction: str = Field(min_length=3)
    citations: list[int] = Field(min_length=1)


class WorkOrderDraft(BaseModel):
    title: str = Field(min_length=3)
    equipment: str
    steps: list[Step] = Field(min_length=1)
    warnings: list[str] = []

    @field_validator("steps")
    @classmethod
    def _steps_have_text(cls, steps: list[Step]) -> list[Step]:
        if any(not s.instruction.strip() for s in steps):
            raise ValueError("every step needs an instruction")
        return steps


class ApprovalError(RuntimeError):
    pass


class WorkOrderStore:
    """JSON-file store; one file per draft, so every decision is auditable."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, wo_id: str) -> Path:
        return self.root / f"{wo_id}.json"

    def propose(self, draft: WorkOrderDraft, pages_read: set[int], question: str) -> str:
        cited = {c for s in draft.steps for c in s.citations}
        unread = sorted(cited - pages_read)
        if unread:
            raise ApprovalError(f"draft cites pages the agent did not read: {unread}")
        wo_id = uuid.uuid4().hex[:12]
        record = {
            "id": wo_id,
            "status": "pending_approval",
            "question": question,
            "draft": draft.model_dump(),
            "proposed_at": datetime.now(timezone.utc).isoformat(),
            "decision": None,
        }
        self._path(wo_id).write_text(json.dumps(record, indent=1))
        return wo_id

    def get(self, wo_id: str) -> dict:
        path = self._path(wo_id)
        if not path.exists():
            raise ApprovalError(f"unknown work order {wo_id}")
        return json.loads(path.read_text())

    def _decide(self, wo_id: str, status: Status, approver: str, reason: str) -> dict:
        if not approver.strip():
            raise ApprovalError("a named approver is required")
        record = self.get(wo_id)
        if record["status"] != "pending_approval":
            raise ApprovalError(f"work order {wo_id} is already {record['status']}")
        record["status"] = status
        record["decision"] = {
            "by": approver,
            "at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }
        self._path(wo_id).write_text(json.dumps(record, indent=1))
        return record

    def approve(self, wo_id: str, approver: str, reason: str = "") -> dict:
        return self._decide(wo_id, "approved", approver, reason)

    def reject(self, wo_id: str, approver: str, reason: str) -> dict:
        return self._decide(wo_id, "rejected", approver, reason)

    def created(self) -> list[dict]:
        """Only approved drafts count as work orders."""
        records = [json.loads(p.read_text()) for p in sorted(self.root.glob("*.json"))]
        return [r for r in records if r["status"] == "approved"]
