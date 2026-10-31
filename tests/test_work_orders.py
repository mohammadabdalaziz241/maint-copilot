import pytest
from pydantic import ValidationError

from maint_copilot.work_orders import ApprovalError, Step, WorkOrderDraft, WorkOrderStore


def draft(pages=(2,)):
    return WorkOrderDraft(
        title="Replace flange gasket",
        equipment="R-11 refueler",
        steps=[Step(instruction="Torque flange bolts to 35 in-lb", citations=list(pages))],
        warnings=["Ground the vehicle first"],
    )


def test_nothing_is_created_without_approval(tmp_path):
    store = WorkOrderStore(tmp_path)
    wo = store.propose(draft(), pages_read={2}, question="q")
    assert store.get(wo)["status"] == "pending_approval"
    assert store.created() == []
    store.approve(wo, approver="M. Abdalaziz")
    created = store.created()
    assert len(created) == 1 and created[0]["decision"]["by"] == "M. Abdalaziz"


def test_draft_cannot_cite_unread_pages(tmp_path):
    with pytest.raises(ApprovalError, match="did not read"):
        WorkOrderStore(tmp_path).propose(draft(pages=(2, 9)), pages_read={2}, question="q")


def test_decisions_are_final_and_need_a_named_approver(tmp_path):
    store = WorkOrderStore(tmp_path)
    wo = store.propose(draft(), pages_read={2}, question="q")
    with pytest.raises(ApprovalError, match="named approver"):
        store.approve(wo, approver=" ")
    store.reject(wo, approver="Supervisor", reason="wrong equipment")
    with pytest.raises(ApprovalError, match="already rejected"):
        store.approve(wo, approver="Supervisor")
    assert store.created() == []


def test_steps_must_cite_a_page():
    with pytest.raises(ValidationError):
        Step(instruction="Do it", citations=[])
