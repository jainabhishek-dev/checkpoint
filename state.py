"""Global in-memory state: caches for workflows, checkpoints, admins, active jobs."""

import asyncio
import tempfile
from pathlib import Path

import db

# ── Constants ─────────────────────────────────────────────────────────────────

SUPER_ADMIN = "abhishek.jain@leadschool.in"

# Temp directory for per-job files (page images, findings.json, job.json)
_JOBS_DIR = Path(tempfile.gettempdir()) / "checkpoint_jobs"
_JOBS_DIR.mkdir(exist_ok=True)

# Review-run job IDs with a background processing task currently running in THIS
# process. Guards against double-launching the same run (e.g. a double-click, or
# Resume racing an already-live task). This is process-local, in-memory state —
# it assumes a single backend instance/worker (matches render.yaml's --workers 1).
# If the service is ever scaled to multiple instances, this guard (and the
# subscriber queues below) would need to move to something shared like Redis.
_ACTIVE_JOBS: set[str] = set()

# Live-view fan-out for review runs: each SSE connection that attaches while a
# run is still processing gets its own queue, fed by the background task via
# publish(). Processing itself never depends on a queue existing — closing every
# browser tab watching a run does not stop it; queues are purely for live viewers.
_JOB_SUBSCRIBERS: dict[str, list["asyncio.Queue"]] = {}


def subscribe(job_id: str) -> "asyncio.Queue":
    q: asyncio.Queue = asyncio.Queue()
    _JOB_SUBSCRIBERS.setdefault(job_id, []).append(q)
    return q


def unsubscribe(job_id: str, q: "asyncio.Queue") -> None:
    subs = _JOB_SUBSCRIBERS.get(job_id)
    if not subs:
        return
    if q in subs:
        subs.remove(q)
    if not subs:
        _JOB_SUBSCRIBERS.pop(job_id, None)


def publish(job_id: str, event: str, data: dict) -> None:
    """Fan an event out to every live SSE viewer currently attached to job_id.
    No-op if nobody is watching — the background task never blocks on this."""
    for q in _JOB_SUBSCRIBERS.get(job_id, []):
        q.put_nowait((event, data))


# ── In-memory caches (populated at startup, refreshed after each mutation) ────

WORKFLOWS: list[dict] = []
CHECKPOINTS: list[dict] = []
CHECKPOINT_MAP: dict[str, dict] = {}
CATEGORIES: dict[str, list[dict]] = {}
ADMINS: set[str] = set()


# ── Reload helpers ─────────────────────────────────────────────────────────────

def _group_by_cat(checkpoints: list[dict]) -> dict[str, list[dict]]:
    """Internal — avoids importing utils (which imports state) and causing a cycle."""
    result: dict[str, list[dict]] = {}
    for cp in checkpoints:
        result.setdefault(cp["category"], []).append(cp)
    return result


def reload_checkpoints() -> None:
    global CHECKPOINTS, CHECKPOINT_MAP, CATEGORIES
    CHECKPOINTS = db.fetch_all_checkpoints()
    CHECKPOINT_MAP = {cp["id"]: cp for cp in CHECKPOINTS}
    CATEGORIES = _group_by_cat(CHECKPOINTS)


def reload_workflows() -> None:
    global WORKFLOWS
    WORKFLOWS = db.fetch_all_workflows()


def reload_admins() -> None:
    global ADMINS
    ADMINS = {a["email"] for a in db.fetch_all_admins()}


# ── Role helpers ───────────────────────────────────────────────────────────────

def is_super_admin(user: dict | None) -> bool:
    return bool(user) and user.get("email") == SUPER_ADMIN


def is_admin(user: dict | None) -> bool:
    return bool(user) and (
        user.get("email") == SUPER_ADMIN or user.get("email") in ADMINS
    )


# ── Populate caches at import time ────────────────────────────────────────────
reload_workflows()
reload_checkpoints()
reload_admins()
