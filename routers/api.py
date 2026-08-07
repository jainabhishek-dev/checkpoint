"""JSON API router — all /api/* endpoints consumed by the React SPA.

Every endpoint returns JSON. Auth is enforced via the shared session cookie
(set by /login/google → /auth/callback on the HTML side).
"""

import asyncio
import json
import re
import uuid
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Body, Form, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import Request

import auth
import db
import state
from utils import (
    ctx, ensure_job_dir, filter_by_workflow, group_by_category,
    load_job, next_checkpoint_id, save_job, slugify, to_ist,
    cascade_delete_workflow,
)
from services.drive_service import get_file_as_pdf, get_pdf_bytes_by_id, fetch_drive_comments_with_pages
from services.ak_ai import AK_REVIEW_DEFAULT_PROMPT, list_exercises, extract_exercise_questions, verify_exercise_questions, review_ak_exercise
from services.review_run_engine import create_run, resume_run
from services.history_saver import save_ak_run_to_history

router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_user(request: Request) -> dict:
    user = auth.get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _require_token(request: Request) -> dict:
    token = auth.get_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return token


def _require_admin(request: Request) -> dict:
    user = _require_user(request)
    if not state.is_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def _require_super_admin(request: Request) -> dict:
    user = _require_user(request)
    if not state.is_super_admin(user):
        raise HTTPException(status_code=403, detail="Super admin access required")
    return user


# ── Auth ───────────────────────────────────────────────────────────────────────

@router.get("/auth/me")
async def get_me(request: Request):
    """Return the current user's profile and role flags, or 401."""
    user = auth.get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "user": user,
        "is_admin": state.is_admin(user),
        "is_super_admin": state.is_super_admin(user),
    }


# ── Drive image proxy ─────────────────────────────────────────────────────────

@router.get("/drive-image/{file_id}")
async def proxy_drive_image(request: Request, file_id: str):
    """Proxy a Drive page image using an authenticated httpx fetch.

    LEAD School's Google Workspace blocks unauthenticated Drive access at the
    domain level — even 'anyone with the link' files redirect to sign-in for
    anonymous requests. Uses the user's OAuth Bearer token with the Drive REST
    API directly (no google-api-python-client Resource object, ~15 MB overhead
    eliminated vs the old proxy). Works for both @leadschool.in and @lbpl.co.in.
    """
    token = auth.get_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    import httpx
    try:
        access_token = token.get("access_token", "")
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.get(
                f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Drive fetch failed: {e}")
    content_type = resp.headers.get("content-type", "unknown")
    is_html = "text/html" in content_type
    print(f"[drive-image] id={file_id} status={resp.status_code} type={content_type} size={len(resp.content)} is_html={is_html}")
    return Response(content=resp.content, media_type="image/jpeg", headers={
        "Cache-Control": "private, max-age=3600",
    })


# ── Workflows & checkpoints ────────────────────────────────────────────────────

@router.get("/workflows")
async def get_workflows(request: Request):
    _require_user(request)
    return {
        "workflows": state.WORKFLOWS,
        "review_workflows": [w for w in state.WORKFLOWS if w.get("type", "review") == "review"],
        "cic_workflows": [w for w in state.WORKFLOWS if w.get("type") == "cic"],
        "ak_workflows": [w for w in state.WORKFLOWS if w.get("type") == "ak_review"],
    }


@router.get("/checkpoints")
async def get_checkpoints(request: Request, workflow_id: str = ""):
    _require_user(request)
    if workflow_id:
        filtered = filter_by_workflow(state.CHECKPOINTS, workflow_id)
    else:
        filtered = state.CHECKPOINTS
    return {
        "checkpoints": filtered,
        "categories": group_by_category(filtered),
    }


# ── Review job ─────────────────────────────────────────────────────────────────

@router.post("/preview-prompt")
async def api_preview_prompt(request: Request, body: dict = Body(...)):
    """Build and return the default page and document prompts for a set of checkpoints."""
    _require_user(request)
    from services.review_ai import _build_vision_prompt, _build_document_prompt
    checkpoint_ids: list[str] = body.get("checkpoint_ids") or []
    workflow_id = (body.get("workflow_id") or "").strip()
    wf = next((w for w in state.WORKFLOWS if w["id"] == workflow_id), {})
    checkpoints = [state.CHECKPOINT_MAP[cid] for cid in checkpoint_ids if cid in state.CHECKPOINT_MAP]
    page_cps = [cp for cp in checkpoints if cp.get("scope") != "document"]
    doc_cps = [cp for cp in checkpoints if cp.get("scope") == "document"]
    return {
        "page_prompt": _build_vision_prompt(page_cps, "{page_num}", wf.get("name", "")) if page_cps else "",
        "doc_prompt": _build_document_prompt(doc_cps) if doc_cps else "",
    }


@router.post("/check")
async def api_run_check(request: Request, body: dict = Body(...)):
    """Validate input, create and start a review run. Returns {job_id}."""
    user = _require_user(request)
    token = _require_token(request)

    drive_url = (body.get("drive_url") or "").strip()
    workflow_id = (body.get("workflow_id") or "").strip()
    checkpoint_ids: list[str] = body.get("checkpoint_ids") or []
    custom_page_prompt: str | None = body.get("custom_page_prompt") or None
    custom_doc_prompt: str | None = body.get("custom_doc_prompt") or None

    if not workflow_id or workflow_id not in [w["id"] for w in state.WORKFLOWS]:
        raise HTTPException(status_code=400, detail="Please select a valid workflow.")
    if not checkpoint_ids:
        raise HTTPException(status_code=400, detail="Please select at least one checkpoint.")
    if not drive_url:
        raise HTTPException(status_code=400, detail="Please enter a Google Drive file URL.")

    selected_checkpoints = [
        state.CHECKPOINT_MAP[cid] for cid in checkpoint_ids if cid in state.CHECKPOINT_MAP
    ]
    workflow = next((w for w in state.WORKFLOWS if w["id"] == workflow_id), {})

    try:
        result = await create_run(
            token=token,
            checked_by=user["email"],
            drive_url=drive_url,
            workflow=workflow,
            selected_checkpoints=selected_checkpoints,
            custom_page_prompt=custom_page_prompt,
            custom_doc_prompt=custom_doc_prompt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        err = str(exc)
        if "invalid_grant" in err or "Token has been expired" in err:
            raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
        raise HTTPException(status_code=400, detail=f"Could not read the file: {err}")

    return result


@router.post("/retry-check/{job_id}")
async def api_retry_check(request: Request, job_id: str):
    _require_user(request)
    token = _require_token(request)
    try:
        return await resume_run(job_id, token)
    except LookupError:
        raise HTTPException(status_code=404, detail="Job not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/insert-comments/{job_id}")
async def api_insert_comments(request: Request, job_id: str, body: dict = Body(...)):
    """Post selected findings as Drive comments. Body: {finding_ids: [int, ...]}"""
    _require_user(request)
    token = _require_token(request)

    from services.drive_service import post_selected_comments, extract_file_id

    run = db.fetch_run(job_id)
    if not run:
        raise HTTPException(status_code=404, detail="Job not found")

    all_findings = db.fetch_run_findings(job_id)
    finding_ids = [int(fid) for fid in (body.get("finding_ids") or []) if str(fid).isdigit()]
    selected = [f for f in all_findings if f.get("id") in finding_ids]

    file_data = {
        "file_id": extract_file_id(run["drive_url"]),
        "file_type": run["file_type"],
        "title": run["document_name"],
    }
    loop = asyncio.get_running_loop()
    try:
        posted = await loop.run_in_executor(
            None, partial(post_selected_comments, token, file_data, selected, state.CHECKPOINT_MAP)
        )
        return {"posted": posted, "total_selected": len(selected)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── CIC job ────────────────────────────────────────────────────────────────────

@router.post("/cic-check")
async def api_run_cic_check(request: Request, body: dict = Body(...)):
    """Validate CIC inputs, fetch comments, create job. Returns {job_id}."""
    user = _require_user(request)
    token = _require_token(request)

    workflow_id = (body.get("workflow_id") or "").strip()
    commented_url = (body.get("commented_url") or "").strip()
    revised_url = (body.get("revised_url") or "").strip()

    workflow = next((w for w in state.WORKFLOWS if w["id"] == workflow_id), None)
    if not workflow or workflow.get("type") != "cic":
        raise HTTPException(status_code=400, detail="Invalid CIC workflow.")
    if not commented_url or not revised_url:
        raise HTTPException(status_code=400, detail="Both file URLs are required.")

    loop = asyncio.get_running_loop()

    try:
        f1_data = await loop.run_in_executor(None, partial(get_file_as_pdf, token, commented_url))
    except Exception as exc:
        err = str(exc)
        if "invalid_grant" in err or "Token has been expired" in err:
            raise HTTPException(status_code=401, detail="Session expired.")
        raise HTTPException(status_code=400, detail=f"Commented file error: {err}")

    try:
        f2_data = await loop.run_in_executor(None, partial(get_file_as_pdf, token, revised_url))
    except Exception as exc:
        err = str(exc)
        if "invalid_grant" in err or "Token has been expired" in err:
            raise HTTPException(status_code=401, detail="Session expired.")
        raise HTTPException(status_code=400, detail=f"Revised file error: {err}")

    if f1_data["file_type"] != "pdf":
        raise HTTPException(status_code=400, detail="Commented file must be a PDF.")
    if f2_data["file_type"] != "pdf":
        raise HTTPException(status_code=400, detail="Revised file must be a PDF.")

    try:
        comments = await loop.run_in_executor(
            None, partial(fetch_drive_comments_with_pages, token, f1_data["file_id"])
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch comments: {exc}")

    if not comments:
        raise HTTPException(status_code=400, detail="No unresolved comments found on the commented file.")

    job_id = uuid.uuid4().hex
    save_job(job_id, {
        "job_type": "cic",
        "workflow_id": workflow_id,
        "workflow_name": workflow["name"],
        "checked_by": user["email"],
        "commented_file_id": f1_data["file_id"],
        "commented_file_title": f1_data["title"],
        "commented_drive_url": commented_url,
        "revised_file_id": f2_data["file_id"],
        "revised_file_title": f2_data["title"],
        "revised_drive_url": revised_url,
        "comments": comments,
        "status": "processing",
    })
    return {
        "job_id": job_id,
        "commented_title": f1_data["title"],
        "revised_title": f2_data["title"],
        "total_comments": len(comments),
    }


# ── History ────────────────────────────────────────────────────────────────────

@router.get("/history")
async def api_get_history(request: Request, tab: str = "review", workflow: str = ""):
    _require_user(request)
    workflow_id = workflow or None
    review_workflows = [w for w in state.WORKFLOWS if w.get("type", "review") == "review"]
    cic_workflows = [w for w in state.WORKFLOWS if w.get("type") == "cic"]
    ak_workflows = [w for w in state.WORKFLOWS if w.get("type") == "ak_review"]
    if tab == "cic":
        cic_runs = db.fetch_cic_runs(workflow_id=workflow_id)
        for r in cic_runs:
            r["created_at"] = to_ist(r.get("created_at"))
        return {
            "runs": [], "cic_runs": cic_runs, "ak_runs": [],
            "active_tab": "cic",
            "review_workflows": review_workflows,
            "cic_workflows": cic_workflows,
            "ak_workflows": ak_workflows,
        }
    elif tab == "ak":
        ak_runs = db.fetch_ak_runs(workflow_id=workflow_id)
        for r in ak_runs:
            r["created_at"] = to_ist(r.get("created_at"))
        return {
            "runs": [], "cic_runs": [], "ak_runs": ak_runs,
            "active_tab": "ak",
            "review_workflows": review_workflows,
            "cic_workflows": cic_workflows,
            "ak_workflows": ak_workflows,
        }
    else:
        runs = db.fetch_runs(workflow_id=workflow_id)
        for r in runs:
            r["created_at"] = to_ist(r.get("created_at"))
        return {
            "runs": runs, "cic_runs": [], "ak_runs": [],
            "active_tab": "review",
            "review_workflows": review_workflows,
            "cic_workflows": cic_workflows,
            "ak_workflows": ak_workflows,
        }


@router.get("/history/cic/{run_id}")
async def api_get_cic_run(request: Request, run_id: str):
    _require_user(request)
    run = db.fetch_cic_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="CIC run not found")
    run["created_at"] = to_ist(run.get("created_at"))

    pages = db.fetch_cic_run_pages(run_id)
    comments = db.fetch_cic_comments(run_id)

    f1_image_map: dict[int, str] = {}
    f2_image_map: dict[int, str] = {}
    for p in pages:
        if p["file_version"] == "commented":
            f1_image_map[p["page_num"]] = p["drive_file_id"]
        else:
            f2_image_map[p["page_num"]] = p["drive_file_id"]

    page_comments_map: dict[str, list] = {}
    global_comments = []
    for c in comments:
        op = c.get("original_page")
        if op is not None:
            page_comments_map.setdefault(str(op), []).append(c)
        else:
            global_comments.append(c)

    return {
        "run": run,
        "total_pages": run.get("total_pages", 0),
        "f1_image_map": {str(k): v for k, v in f1_image_map.items()},
        "f2_image_map": {str(k): v for k, v in f2_image_map.items()},
        "page_comments_map": page_comments_map,
        "global_comments": global_comments,
        "needs_images": len(f1_image_map) < run.get("total_pages", 0) or len(f2_image_map) < run.get("total_pages", 0),
    }


@router.get("/history/cic/{run_id}/pages")
async def api_get_cic_run_pages(request: Request, run_id: str):
    _require_user(request)
    pages = db.fetch_cic_run_pages(run_id)
    return {"pages": pages}


@router.get("/history/ak/{run_id}")
async def api_get_ak_run(request: Request, run_id: str):
    _require_user(request)
    run = db.fetch_ak_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="AK run not found")
    run["created_at"] = to_ist(run.get("created_at"))
    questions = db.fetch_ak_question_results(run_id)
    return {"run": run, "questions": questions}


@router.get("/history/{run_id}")
async def api_get_run(request: Request, run_id: str):
    _require_user(request)
    run = db.fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    run["created_at"] = to_ist(run.get("created_at"))

    pages = db.fetch_run_pages(run_id)
    findings = db.fetch_run_findings(run_id)

    page_findings: dict[str, list] = {}
    doc_findings: list = []
    for f in findings:
        if f["page_num"] is None:
            doc_findings.append(f)
        else:
            page_findings.setdefault(str(f["page_num"]), []).append(f)

    checkpoint_map = {cp["id"]: cp["category"] for cp in state.CHECKPOINTS}
    page_image_map = {str(p["page_num"]): p["drive_file_id"] for p in pages}

    return {
        "run": run,
        "page_findings": page_findings,
        "doc_findings": doc_findings,
        "checkpoint_map": checkpoint_map,
        "page_image_map": page_image_map,
        "total_pages": run["total_pages"],
    }


@router.post("/findings/{finding_id}/review")
async def api_update_finding_review(request: Request, finding_id: str, body: dict = Body(...)):
    _require_user(request)
    review_status = (body.get("review_status") or "").strip()
    review_comment = (body.get("review_comment") or "").strip()
    if review_status not in ("valid", "invalid"):
        raise HTTPException(status_code=400, detail="review_status must be 'valid' or 'invalid'")
    if not review_comment:
        raise HTTPException(status_code=400, detail="review_comment is required")
    try:
        db.update_finding_review(finding_id, review_status, review_comment)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── AK Review ────────────────────────────────────────────────────────────────

@router.get("/ak-default-prompt")
async def api_ak_default_prompt(request: Request):
    """Return the default AK review prompt for pre-filling the prompt editor."""
    _require_user(request)
    return {"prompt": AK_REVIEW_DEFAULT_PROMPT}


@router.post("/ak-check")
async def api_start_ak_job(request: Request):
    """Validate AK Review inputs, create job, return job_id."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)
    if not user or not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    body = await request.json()
    workflow_id = body.get("workflow_id", "")
    chapter_url = (body.get("chapter_url") or "").strip()
    ak_url = (body.get("ak_url") or "").strip()
    custom_prompt = body.get("custom_prompt") or None

    workflow = next((w for w in state.WORKFLOWS if w["id"] == workflow_id), None)
    if not workflow or workflow.get("type") != "ak_review":
        raise HTTPException(status_code=400, detail="Invalid AK Review workflow.")
    if not chapter_url or not ak_url:
        raise HTTPException(status_code=400, detail="Both chapter URL and answer key URL are required.")

    loop = asyncio.get_running_loop()

    try:
        chapter_data = await loop.run_in_executor(None, partial(get_file_as_pdf, token, chapter_url))
    except Exception as exc:
        err = str(exc)
        if "invalid_grant" in err or "Token has been expired" in err:
            raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
        raise HTTPException(status_code=400, detail=f"Chapter file error: {err}")

    try:
        ak_data = await loop.run_in_executor(None, partial(get_file_as_pdf, token, ak_url))
    except Exception as exc:
        err = str(exc)
        if "invalid_grant" in err or "Token has been expired" in err:
            raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
        raise HTTPException(status_code=400, detail=f"Answer key file error: {err}")

    if chapter_data["file_type"] != "pdf":
        raise HTTPException(status_code=400, detail="Chapter file must be a PDF.")
    if ak_data["file_type"] != "pdf":
        raise HTTPException(status_code=400, detail="Answer key file must be a PDF.")

    job_id = uuid.uuid4().hex
    save_job(job_id, {
        "job_type": "ak_review",
        "workflow_id": workflow_id,
        "workflow_name": workflow["name"],
        "checked_by": user["email"],
        "chapter_file_id": chapter_data["file_id"],
        "chapter_file_title": chapter_data["title"],
        "chapter_drive_url": chapter_url,
        "ak_file_id": ak_data["file_id"],
        "ak_file_title": ak_data["title"],
        "ak_drive_url": ak_url,
        "prompt": custom_prompt,
        "status": "processing",
    })

    return {
        "job_id": job_id,
        "chapter_title": chapter_data["title"],
        "ak_title": ak_data["title"],
    }


def _sort_key_exercise(name: str) -> tuple:
    """Sort key for exercise names: Exercise 5A < 5B < ... < Additional Exercise."""
    import re as _re
    name_lower = name.lower()
    if 'additional' in name_lower:
        return (1, 0, name_lower)
    m = _re.search(r'(\d+)\s*([a-zA-Z]*)', name)
    if m:
        return (0, int(m.group(1)), m.group(2).lower())
    return (0, 999, name_lower)


def _sort_key_question(qno: str) -> str:
    """Natural sort key for question numbers: 1 < 1(a) < 1(b) < 2 < 2(a) < 10."""
    import re as _re
    return _re.sub(r'\d+', lambda m: m.group().zfill(4), (qno or '').lower())


async def _stream_ak_processing(job_id: str, token: dict):
    """SSE generator for AK Review.

    Three visible phases:
      Phase 1 (scanning):   list exercises  → ak_exercises_found
      Phase 2 (extracting): 1 call/exercise → ak_exercise_extracted per exercise
      Phase 3 (reviewing):  1 call/exercise → ak_question per question
    """
    job = load_job(job_id)
    if not job:
        yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
        return

    state._ACTIVE_JOBS.add(job_id)
    try:
        loop = asyncio.get_running_loop()

        # Load both PDFs
        try:
            chapter_pdf = await loop.run_in_executor(
                None, partial(get_pdf_bytes_by_id, token, job["chapter_file_id"])
            )
            chapter_bytes = chapter_pdf["pdf_bytes"]
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': f'Could not load chapter file: {str(e)}'})}\n\n"
            return

        try:
            ak_pdf = await loop.run_in_executor(
                None, partial(get_pdf_bytes_by_id, token, job["ak_file_id"])
            )
            ak_bytes = ak_pdf["pdf_bytes"]
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': f'Could not load answer key file: {str(e)}'})}\n\n"
            return

        # ── Phase 1: List exercises ──────────────────────────────────────────
        try:
            exercise_names = await loop.run_in_executor(
                None, partial(list_exercises, chapter_bytes)
            )
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': f'Could not list exercises: {str(e)}'})}\n\n"
            return

        if not exercise_names:
            yield f"event: error\ndata: {json.dumps({'message': 'No exercises found in the chapter.'})}\n\n"
            return

        # Normalize "Exercise 5 A" → "Exercise 5A" (AI sometimes adds a space)
        exercise_names = [re.sub(r'(\d)\s+([A-Za-z])', r'\1\2', name) for name in exercise_names]
        # Sort into correct document order
        exercise_names = sorted(exercise_names, key=_sort_key_exercise)

        yield f"event: ak_exercises_found\ndata: {json.dumps({'exercises': exercise_names})}\n\n"

        # ── Phase 2: Extract + verify questions per exercise ─────────────────
        _MAX_VERIFY_PASSES = 3
        exercise_map: dict[str, list[dict]] = {}
        for exercise_no in exercise_names:
            try:
                questions = await loop.run_in_executor(
                    None, partial(extract_exercise_questions, chapter_bytes, exercise_no)
                )
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Extraction failed for {exercise_no}: {str(e)}'})}\n\n"
                return

            # Verification loop: keep checking until nothing is missing
            for _pass in range(_MAX_VERIFY_PASSES):
                try:
                    missed = await loop.run_in_executor(
                        None, partial(verify_exercise_questions, chapter_bytes, exercise_no, questions)
                    )
                except Exception:
                    break  # verification failed — proceed with what we have
                seen_qnos = {q.get("question_no", "") for q in questions}
                new_qs = [q for q in missed if q.get("question_no", "") not in seen_qnos]
                if not new_qs:
                    break  # converged — nothing missed
                questions = questions + new_qs

            questions = sorted(questions, key=lambda q: _sort_key_question(q.get("question_no", "")))
            exercise_map[exercise_no] = questions
            yield f"event: ak_exercise_extracted\ndata: {json.dumps({'exercise_no': exercise_no, 'question_count': len(questions), 'questions': questions})}\n\n"

        total_questions = sum(len(qs) for qs in exercise_map.values())
        if total_questions == 0:
            yield f"event: error\ndata: {json.dumps({'message': 'No questions found in any exercise.'})}\n\n"
            return

        yield f"event: ak_start\ndata: {json.dumps({'exercises': exercise_names, 'total_questions': total_questions})}\n\n"

        # ── Phase 3: Review each exercise ────────────────────────────────────
        review_prompt = job.get("prompt") or AK_REVIEW_DEFAULT_PROMPT
        all_results: list[dict] = []

        for exercise_no in exercise_names:
            questions = exercise_map[exercise_no]
            if not questions:
                continue
            yield f"event: ak_exercise_start\ndata: {json.dumps({'exercise_no': exercise_no, 'question_count': len(questions)})}\n\n"

            try:
                results = await loop.run_in_executor(
                    None, partial(review_ak_exercise, chapter_bytes, ak_bytes, exercise_no, questions, review_prompt)
                )
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Review failed for {exercise_no}: {str(e)}'})}\n\n"
                return

            for r in results:
                all_results.append(r)
                yield f"event: ak_question\ndata: {json.dumps(r)}\n\n"

            yield f"event: ak_exercise_done\ndata: {json.dumps({'exercise_no': exercise_no})}\n\n"

        # Compute summary counts
        total = len(all_results)
        present = sum(1 for r in all_results if r.get("present_in_ak") == "Yes")
        missing = sum(1 for r in all_results if r.get("present_in_ak") == "No")
        incorrect = sum(1 for r in all_results if r.get("answer_correct") == "No")
        manual = sum(1 for r in all_results if r.get("answer_correct") == "Manual Review Required")

        asyncio.create_task(save_ak_run_to_history(
            job_id=job_id,
            job=dict(job),
            question_results=list(all_results),
        ))

        yield f"event: ak_done\ndata: {json.dumps({'run_id': job_id, 'total_questions': total, 'present_in_ak': present, 'missing_from_ak': missing, 'incorrect_answers': incorrect, 'manual_review_cases': manual})}\n\n"

        job["status"] = "completed"
        save_job(job_id, job)

    finally:
        state._ACTIVE_JOBS.discard(job_id)


@router.get("/ak-stream/{job_id}")
async def stream_ak_processing(request: Request, job_id: str):
    """SSE endpoint for AK Review job streaming."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)
    if not user or not token:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    if job_id in state._ACTIVE_JOBS:
        async def _already_running():
            yield f"event: error\ndata: {json.dumps({'message': 'This job is already being processed.'})}\n\n"
        return StreamingResponse(
            _already_running(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        _stream_ak_processing(job_id, token),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Admin — workflows ──────────────────────────────────────────────────────────

@router.get("/admin/workflows")
async def api_admin_get_workflows(request: Request):
    _require_admin(request)
    checkpoint_counts: dict[str, int] = {}
    for cp in state.CHECKPOINTS:
        for wf_id in cp.get("workflows", []):
            checkpoint_counts[wf_id] = checkpoint_counts.get(wf_id, 0) + 1
    return {"workflows": state.WORKFLOWS, "checkpoint_counts": checkpoint_counts}


@router.post("/admin/fix-image-permissions")
async def api_fix_image_permissions(request: Request):
    """One-time migration: set public read permission on all existing page images.

    Run this once to backfill permissions on Drive files uploaded before
    upload_jpeg_to_drive started setting permissions automatically.
    """
    _require_admin(request)
    token = _require_token(request)
    from services.drive_service import _build_credentials, _build_drive_service

    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)

    file_ids = list(set(
        db.fetch_all_run_page_image_ids() +
        db.fetch_all_cic_page_image_ids()
    ))

    ok, failed = 0, []
    for fid in file_ids:
        try:
            drive_service.permissions().create(
                fileId=fid,
                body={"type": "anyone", "role": "reader"},
            ).execute()
            ok += 1
        except Exception as e:
            failed.append({"file_id": fid, "error": str(e)})

    return {"fixed": ok, "failed": len(failed), "failed_ids": failed, "total": len(file_ids)}


@router.post("/admin/workflows/add")
async def api_admin_add_workflow(request: Request, body: dict = Body(...)):
    user = _require_admin(request)
    name = (body.get("name") or "").strip()
    description = (body.get("description") or "").strip()
    ai_notes = (body.get("ai_notes") or "").strip()
    action = body.get("action", "manual")
    wf_type = body.get("type", "review")

    if not name:
        raise HTTPException(status_code=400, detail="Workflow name is required.")

    wf_id = slugify(name)
    if not wf_id:
        raise HTTPException(status_code=400, detail="Could not generate a valid ID from that name.")
    if any(w["id"] == wf_id for w in state.WORKFLOWS):
        raise HTTPException(status_code=409, detail=f'A workflow with id "{wf_id}" already exists.')

    sort_order = max((w.get("sort_order", 0) for w in state.WORKFLOWS), default=0) + 1

    if action == "generate":
        if not ai_notes:
            raise HTTPException(status_code=400, detail="AI notes are required for AI generation.")
        from services.workflow_gen import generate_workflow_content
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, partial(generate_workflow_content, name, ai_notes))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        db.insert_workflow({"id": wf_id, "name": name, "description": description,
                            "sort_order": sort_order, "created_by": user["email"],
                            "type": wf_type if wf_type in ("review", "cic", "ak_review") else "review"})

        existing_nums = [
            int(cp["id"].split("_")[1])
            for cp in state.CHECKPOINTS
            if cp["id"].startswith("cp_") and cp["id"].split("_")[1].isdigit()
        ]
        next_num = max(existing_nums, default=0) + 1
        next_sort = max((cp.get("sort_order", 0) for cp in state.CHECKPOINTS), default=0) + 1
        for cp_data in result.get("checkpoints", []):
            try:
                db.insert_checkpoint({"id": f"cp_{next_num:03d}", "category": cp_data["category"].strip(),
                                      "instructions": cp_data["instructions"].strip(),
                                      "type": cp_data["type"], "scope": cp_data["scope"],
                                      "workflows": [wf_id], "sort_order": next_sort})
                next_num += 1
                next_sort += 1
            except Exception:
                pass
        state.reload_checkpoints()
        state.reload_workflows()
        return {"ok": True, "workflow_id": wf_id, "checkpoint_count": len(result.get("checkpoints", []))}

    db.insert_workflow({"id": wf_id, "name": name, "description": description,
                        "sort_order": sort_order, "created_by": user["email"],
                        "type": wf_type if wf_type in ("review", "cic", "ak_review") else "review"})
    state.reload_workflows()
    return {"ok": True, "workflow_id": wf_id}


@router.post("/admin/workflows/{wf_id}/edit")
async def api_admin_edit_workflow(request: Request, wf_id: str, body: dict = Body(...)):
    _require_admin(request)
    name = (body.get("name") or "").strip()
    description = (body.get("description") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Workflow name is required.")
    try:
        db.update_workflow(wf_id, {"name": name, "description": description})
        state.reload_workflows()
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/admin/workflows/{wf_id}/delete")
async def api_admin_delete_workflow(request: Request, wf_id: str):
    user = _require_admin(request)
    workflow = next((w for w in state.WORKFLOWS if w["id"] == wf_id), None)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    if not state.is_super_admin(user) and workflow.get("created_by") != user.get("email"):
        raise HTTPException(status_code=403, detail="Only the super admin or workflow creator can delete this.")
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(cascade_delete_workflow, wf_id))
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Admin — checkpoints ────────────────────────────────────────────────────────

@router.post("/admin/checkpoints/add")
async def api_admin_add_checkpoint(request: Request, body: dict = Body(...)):
    _require_admin(request)
    new_id = next_checkpoint_id()
    sort_order = max((cp["sort_order"] for cp in state.CHECKPOINTS), default=0) + 1
    try:
        db.insert_checkpoint({
            "id": new_id,
            "category": (body.get("category") or "").strip(),
            "instructions": (body.get("instructions") or "").strip(),
            "type": body.get("type", "rule"),
            "scope": body.get("scope", "page"),
            "workflows": body.get("workflows") or [],
            "sort_order": sort_order,
        })
        state.reload_checkpoints()
        return {"ok": True, "checkpoint_id": new_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/admin/checkpoints/{cp_id}/edit")
async def api_admin_edit_checkpoint(request: Request, cp_id: str, body: dict = Body(...)):
    _require_admin(request)
    try:
        db.update_checkpoint(cp_id, {
            "instructions": (body.get("instructions") or "").strip(),
            "type": body.get("type", "rule"),
            "scope": body.get("scope", "page"),
            "workflows": body.get("workflows") or [],
        })
        state.reload_checkpoints()
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/admin/checkpoints/{cp_id}/delete")
async def api_admin_delete_checkpoint(request: Request, cp_id: str):
    _require_admin(request)
    try:
        db.delete_checkpoint(cp_id)
        state.reload_checkpoints()
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Admin — admins ─────────────────────────────────────────────────────────────

@router.get("/admin/admins")
async def api_admin_get_admins(request: Request):
    _require_admin(request)
    return {"admins": db.fetch_all_admins()}


@router.post("/admin/admins/add")
async def api_admin_add_admin(request: Request, body: dict = Body(...)):
    user = _require_admin(request)
    email = (body.get("email") or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")
    try:
        db.insert_admin({"email": email, "added_by": user["email"]})
        state.reload_admins()
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/admin/admins/{email}/delete")
async def api_admin_delete_admin(request: Request, email: str):
    _require_super_admin(request)
    try:
        db.delete_admin(email)
        state.reload_admins()
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
