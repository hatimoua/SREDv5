"""
Gate logic for the world model.

Checks blocking conditions and updates Run status accordingly.
"""
from sqlmodel import Session, select
from sred.models.core import Run, RunStatus
from sred.models.world import (
    Contradiction,
    ContradictionSeverity,
    ContradictionStatus,
    ReviewTask,
    ReviewTaskStatus,
    DecisionLock,
)
from sred.logging import logger


def get_blocking_contradictions(session: Session, run_id: int) -> list[Contradiction]:
    """Return all OPEN + BLOCKING contradictions for a run."""
    return list(session.exec(
        select(Contradiction).where(
            Contradiction.run_id == run_id,
            Contradiction.severity == ContradictionSeverity.BLOCKING,
            Contradiction.status == ContradictionStatus.OPEN,
        )
    ).all())


def get_open_blocking_tasks(session: Session, run_id: int) -> list[ReviewTask]:
    """Return all OPEN review tasks with BLOCKING severity."""
    return list(session.exec(
        select(ReviewTask).where(
            ReviewTask.run_id == run_id,
            ReviewTask.severity == ContradictionSeverity.BLOCKING,
            ReviewTask.status == ReviewTaskStatus.OPEN,
        )
    ).all())


def has_active_lock(session: Session, run_id: int, issue_key: str) -> bool:
    """Check whether an active DecisionLock exists for the given issue_key in this run."""
    lock = session.exec(
        select(DecisionLock).where(
            DecisionLock.run_id == run_id,
            DecisionLock.issue_key == issue_key,
            DecisionLock.active == True,  # noqa: E712
        )
    ).first()
    return lock is not None


def update_run_gate_status(session: Session, run_id: int) -> RunStatus:
    """
    Evaluate blocking conditions and update the Run status.

    Rules:
    - If any OPEN + BLOCKING contradictions or tasks exist -> NEEDS_REVIEW
    - Otherwise leave status unchanged (caller decides PROCESSING / COMPLETED)

    Returns the new status.
    """
    run = session.get(Run, run_id)
    if not run:
        raise ValueError(f"Run {run_id} not found")

    blockers = get_blocking_contradictions(session, run_id)
    blocking_tasks = get_open_blocking_tasks(session, run_id)

    if blockers or blocking_tasks:
        if run.status != RunStatus.NEEDS_REVIEW:
            logger.info(f"Run {run_id} -> NEEDS_REVIEW ({len(blockers)} contradictions, {len(blocking_tasks)} tasks)")
            run.status = RunStatus.NEEDS_REVIEW
            session.add(run)
            session.commit()
        return RunStatus.NEEDS_REVIEW

    # If we were in NEEDS_REVIEW and all blockers are cleared, move back to PROCESSING
    if run.status == RunStatus.NEEDS_REVIEW:
        logger.info(f"Run {run_id} -> PROCESSING (all blockers resolved)")
        run.status = RunStatus.PROCESSING
        session.add(run)
        session.commit()
        return RunStatus.PROCESSING

    return run.status
