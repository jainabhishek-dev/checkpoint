"""Review workflow routes: /check, /process/{id}, /stream/{id}, /job/{id}/page/{n},
/retry-check/{id}, /insert-comments/{id}.

Processing itself lives in services/review_run_engine.py as a background task
independent of any request — these routes only create/resume runs and expose a
read-only SSE view (replay of what's already in Supabase, then a live tail).
"""

import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from starlette.requests import Request

import auth
import db
import state
from utils import ctx, filter_by_workflow, group_by_category, templates
from services.drive_service import extract_file_id, post_selected_comments
from services.review_run_engine import create_run, resume_run

router = APIRouter()


# ── SSE view generator (replay from Supabase, then live tail) ──────────────────

async def _stream_view(job_id: str):
    """Reconstruct a run's progress from Supabase for a newly-attached viewer,
    then — if it's still processing — subscribe to live updates from whatever
    background task (if any) is currently working on it. Never starts or stops
    processing itself; multiple viewers (or a refreshed tab) can attach safely."""
    loop = asyncio.get_running_loop()
    q = None
    try:
        run = await loop.run_in_executor(None, db.fetch_run, job_id)
        if not run:
            yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
            return

        if run.get("status") == "processing":
            # Subscribe before re-fetching so nothing published in the gap is missed.
            q = state.subscribe(job_id)
            run = await loop.run_in_executor(None, db.fetch_run, job_id)

        total_pages = run.get("total_pages") or 0
        title = run.get("document_name", "")
        if total_pages:
            yield f"event: start\ndata: {json.dumps({'total_pages': total_pages, 'title': title})}\n\n"

        pages = await loop.run_in_executor(None, db.fetch_run_pages, job_id)
        findings = await loop.run_in_executor(None, db.fetch_run_findings, job_id)

        image_by_page = {p["page_num"]: p["drive_file_id"] for p in pages}
        findings_by_page: dict[int, list] = {}
        doc_findings: list = []
        for f in findings:
            if f.get("page_num") is None:
                doc_findings.append(f)
            else:
                findings_by_page.setdefault(f["page_num"], []).append(f)

        last_successful_page = run.get("last_successful_page") or 0
        for page_num in range(1, last_successful_page + 1):
            drive_file_id = image_by_page.get(page_num)
            yield f"event: page_ready\ndata: {json.dumps({'page': page_num, 'total_pages': total_pages, 'drive_file_id': drive_file_id})}\n\n"
            yield f"event: page_findings\ndata: {json.dumps({'page': page_num, 'findings': findings_by_page.get(page_num, [])})}\n\n"

        if doc_findings:
            yield f"event: document_findings\ndata: {json.dumps({'findings': doc_findings})}\n\n"

        status = run.get("status")

        if status == "completed":
            yield f"event: all_done\ndata: {json.dumps({'total_findings': run.get('total_findings', 0)})}\n\n"
            return

        if status == "failed":
            yield (
                f"event: partial_complete\ndata: "
                f"{json.dumps({'last_successful_page': last_successful_page, 'total_pages': total_pages, 'error_message': run.get('error_message') or 'Unknown error'})}\n\n"
            )
            return

        # status == "processing" — live-follow via the subscribed queue.
        if q is None:
            q = state.subscribe(job_id)
        terminal = {"all_done", "error", "partial_complete"}
        while True:
            event, data = await q.get()
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
            if event in terminal:
                return
    finally:
        if q is not None:
            state.unsubscribe(job_id, q)


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/check", response_class=HTMLResponse)
async def run_check(
    request: Request,
    drive_url: Annotated[str, Form()],
    workflow_id: Annotated[str, Form()],
    checkpoint_ids: Annotated[list[str], Form()] = [],
):
    """Validate input, create+start a run, then redirect to the processing page."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)

    if not user or not token:
        return RedirectResponse(url="/login", status_code=303)

    if not workflow_id or workflow_id not in [w["id"] for w in state.WORKFLOWS]:
        return templates.TemplateResponse("index.html", ctx(
            request, user,
            workflows=state.WORKFLOWS,
            selected_workflow=None,
            categories={},
            error="Please select a workflow before running the check.",
        ))

    if not checkpoint_ids:
        filtered_checkpoints = filter_by_workflow(state.CHECKPOINTS, workflow_id)
        filtered_categories = group_by_category(filtered_checkpoints)
        selected_workflow = next((w for w in state.WORKFLOWS if w["id"] == workflow_id), None)
        return templates.TemplateResponse("index.html", ctx(
            request, user,
            workflows=state.WORKFLOWS,
            selected_workflow=selected_workflow,
            categories=filtered_categories,
            error="Please select at least one checkpoint before running the check.",
        ))

    if not drive_url.strip():
        filtered_checkpoints = filter_by_workflow(state.CHECKPOINTS, workflow_id)
        filtered_categories = group_by_category(filtered_checkpoints)
        selected_workflow = next((w for w in state.WORKFLOWS if w["id"] == workflow_id), None)
        return templates.TemplateResponse("index.html", ctx(
            request, user,
            workflows=state.WORKFLOWS,
            selected_workflow=selected_workflow,
            categories=filtered_categories,
            error="Please enter a Google Drive file URL.",
        ))

    selected_checkpoints = [
        state.CHECKPOINT_MAP[cid] for cid in checkpoint_ids if cid in state.CHECKPOINT_MAP
    ]
    workflow = next((w for w in state.WORKFLOWS if w["id"] == workflow_id), None)

    try:
        result = await create_run(
            token=token,
            checked_by=user["email"],
            drive_url=drive_url.strip(),
            workflow=workflow,
            selected_checkpoints=selected_checkpoints,
        )
    except ValueError as exc:
        filtered_checkpoints = filter_by_workflow(state.CHECKPOINTS, workflow_id)
        filtered_categories = group_by_category(filtered_checkpoints)
        selected_workflow = next((w for w in state.WORKFLOWS if w["id"] == workflow_id), None)
        return templates.TemplateResponse("index.html", ctx(
            request, user,
            workflows=state.WORKFLOWS,
            selected_workflow=selected_workflow,
            categories=filtered_categories,
            error=str(exc),
        ))
    except Exception as exc:
        error_msg = str(exc)
        if "invalid_grant" in error_msg or "Token has been expired" in error_msg:
            request.session.clear()
            return RedirectResponse(url="/login", status_code=303)
        filtered_checkpoints = filter_by_workflow(state.CHECKPOINTS, workflow_id)
        filtered_categories = group_by_category(filtered_checkpoints)
        selected_workflow = next((w for w in state.WORKFLOWS if w["id"] == workflow_id), None)
        return templates.TemplateResponse("index.html", ctx(
            request, user,
            workflows=state.WORKFLOWS,
            selected_workflow=selected_workflow,
            categories=filtered_categories,
            error=f"Could not read the file: {error_msg}",
        ))

    return RedirectResponse(url=f"/process/{result['job_id']}", status_code=303)


@router.get("/process/{job_id}", response_class=HTMLResponse)
async def show_process(request: Request, job_id: str):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    run = db.fetch_run(job_id)
    if not run:
        return RedirectResponse(url="/?error=Job+not+found.+Please+run+a+new+check.")

    checkpoint_names = {cp["id"]: cp["category"] for cp in state.CHECKPOINTS}

    return templates.TemplateResponse("process.html", ctx(
        request, user,
        job_id=job_id,
        title=run.get("document_name", ""),
        checkpoint_names=checkpoint_names,
    ))


@router.get("/stream/{job_id}")
async def stream_processing(request: Request, job_id: str):
    """SSE endpoint: replays saved progress, then live-tails if still processing."""
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    return StreamingResponse(
        _stream_view(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/job/{job_id}/page/{page_num:int}")
async def serve_page_image(job_id: str, page_num: int):
    """Serve a rendered page image (JPEG) from disk."""
    job_dir = state._JOBS_DIR / job_id
    page_file = job_dir / f"page_{page_num:03d}.jpg"
    if not page_file.exists():
        raise HTTPException(status_code=404, detail="Page image not found")
    return FileResponse(page_file, media_type="image/jpeg")


@router.post("/retry-check/{job_id}")
async def retry_check(request: Request, job_id: str):
    user = auth.get_current_user(request)
    token = auth.get_token(request)

    if not user or not token:
        return RedirectResponse(url="/login", status_code=303)

    try:
        await resume_run(job_id, token)
    except LookupError:
        return RedirectResponse(url="/?error=Job+not+found.", status_code=303)
    except ValueError:
        pass  # already completed — nothing to resume, just show it

    return RedirectResponse(url=f"/process/{job_id}", status_code=303)


@router.post("/insert-comments/{job_id}")
async def insert_comments(
    request: Request,
    job_id: str,
    finding_ids: Annotated[list[str], Form()] = [],
):
    """Insert selected findings as Drive comments."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)

    if not user or not token:
        return RedirectResponse(url="/login", status_code=303)

    run = db.fetch_run(job_id)
    if not run:
        return {"error": "Job not found"}

    all_findings = db.fetch_run_findings(job_id)
    selected_finding_ids = [int(fid) for fid in finding_ids if fid.isdigit()]
    selected_findings = [f for f in all_findings if f.get("id") in selected_finding_ids]

    file_data = {
        "file_id": extract_file_id(run["drive_url"]),
        "file_type": run["file_type"],
        "title": run["document_name"],
    }

    loop = asyncio.get_running_loop()
    try:
        posted = await loop.run_in_executor(
            None, lambda: post_selected_comments(token, file_data, selected_findings, state.CHECKPOINT_MAP)
        )
        return {"posted": posted, "total_selected": len(selected_findings)}
    except Exception as exc:
        return {"error": f"Could not post comments: {str(exc)}"}
