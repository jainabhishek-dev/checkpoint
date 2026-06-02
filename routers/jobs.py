"""Review workflow routes: /check, /process/{id}, /stream/{id}, /job/{id}/page/{n},
/retry-check/{id}, /insert-comments/{id}.
"""

import asyncio
import json
import uuid
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from starlette.requests import Request

import auth
import state
from utils import (
    ctx, ensure_job_dir, filter_by_workflow, group_by_category,
    load_job, save_job, templates,
)
from services.drive_service import (
    get_file_as_pdf, get_pdf_bytes_by_id, post_selected_comments,
)
from services.review_ai import (
    run_vision_check, run_vision_review, run_document_check, run_document_review,
)
from services.history_saver import save_run_to_history

router = APIRouter()


# ── SSE streaming generator ────────────────────────────────────────────────────

async def _stream_processing(job_id: str, token: dict, retry_from: int = None):
    """SSE generator for review processing: renders pages, calls AI, streams results."""
    import fitz
    from io import BytesIO

    job = load_job(job_id)
    if not job:
        yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
        return

    state._ACTIVE_JOBS.add(job_id)
    try:
        workflow_id = job.get("workflow_id", "")
        workflow = next((w for w in state.WORKFLOWS if w["id"] == workflow_id), None)
        if not workflow:
            yield f"event: error\ndata: {json.dumps({'message': f'Workflow \"{workflow_id}\" not found. It may have been deleted.'})}\n\n"
            return
        workflow_name = workflow["name"]

        loop = asyncio.get_running_loop()
        job_dir = ensure_job_dir(job_id)

        # ── Load PDF ──────────────────────────────────────────────────────────
        if retry_from is None:
            try:
                pdf_data = await loop.run_in_executor(
                    None, partial(get_pdf_bytes_by_id, token, job["file_id"])
                )
                pdf_bytes = pdf_data.get("pdf_bytes")
                if not pdf_bytes:
                    raise ValueError("No PDF bytes retrieved")
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Could not export file as PDF: {str(e)}'})}\n\n"
                return

            try:
                pdf_document = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Could not parse PDF: {str(e)}'})}\n\n"
                return

            total_pages = len(pdf_document)
            all_findings = []
            finding_id_counter = 0
            start_page = 1

            yield f"event: start\ndata: {json.dumps({'total_pages': total_pages, 'title': job['title']})}\n\n"
        else:
            try:
                pdf_data = await loop.run_in_executor(
                    None, partial(get_pdf_bytes_by_id, token, job["file_id"])
                )
                pdf_bytes = pdf_data.get("pdf_bytes")
                pdf_document = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Could not re-open PDF: {str(e)}'})}\n\n"
                return

            total_pages = len(pdf_document)
            start_page = retry_from

            findings_file = job_dir / "findings.json"
            if findings_file.exists():
                all_findings = json.loads(findings_file.read_text(encoding="utf-8"))
                finding_id_counter = max([f.get("id", 0) for f in all_findings] + [0]) + 1
            else:
                all_findings = []
                finding_id_counter = 0

            yield f"event: retry_start\ndata: {json.dumps({'starting_page': start_page, 'total_pages': total_pages})}\n\n"

        # Free PDF bytes — fitz has loaded them into its own C-level memory.
        # They will be re-fetched below only if the document-level check is needed.
        del pdf_bytes

        selected_checkpoints = [
            state.CHECKPOINT_MAP[cid] for cid in job["checkpoint_ids"] if cid in state.CHECKPOINT_MAP
        ]

        page_checkpoints = [cp for cp in selected_checkpoints if cp.get("scope") != "document"]
        doc_checkpoints  = [cp for cp in selected_checkpoints if cp.get("scope") == "document"]

        # ── Page-by-page processing (only if page-scope checkpoints selected) ─
        if page_checkpoints:
            for page_num in range(start_page, total_pages + 1):
                try:
                    page = pdf_document[page_num - 1]
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img_bytes = pix.tobytes(output="jpeg")

                    img_path = job_dir / f"page_{page_num:03d}.jpg"
                    img_path.write_bytes(img_bytes)

                    yield f"event: page_ready\ndata: {json.dumps({'page': page_num, 'total_pages': total_pages})}\n\n"

                    findings = await loop.run_in_executor(
                        None, partial(run_vision_check, img_bytes, page_checkpoints, page_num, workflow_name,
                                      job.get("page_prompt") or None)
                    )

                    for finding in findings:
                        finding["id"] = finding_id_counter
                        finding["page_num"] = page_num
                        if "location" not in finding or finding["location"] == "":
                            finding["location"] = f"Page {page_num}"
                        all_findings.append(finding)
                        finding_id_counter += 1

                    yield f"event: page_findings\ndata: {json.dumps({'page': page_num, 'findings': findings})}\n\n"

                    if findings:
                        reviews = await loop.run_in_executor(
                            None, partial(run_vision_review, img_bytes, findings, page_num)
                        )
                        yield f"event: page_review\ndata: {json.dumps({'page': page_num, 'reviews': reviews})}\n\n"

                        review_map = {r["finding_id"]: r for r in reviews}
                        for f in findings:
                            rev = review_map.get(f["id"])
                            if rev:
                                f["review_status"] = rev["verdict"]
                                f["review_comment"] = rev["reason"]

                    del img_bytes

                except Exception as e:
                    job["last_successful_page"] = page_num - 1
                    save_job(job_id, job)
                    try:
                        (job_dir / "findings.json").write_text(
                            json.dumps(all_findings, ensure_ascii=False), encoding="utf-8"
                        )
                    except Exception as save_error:
                        yield f"event: error\ndata: {json.dumps({'message': f'Could not save findings: {str(save_error)}'})}\n\n"

                    yield f"event: partial_complete\ndata: {json.dumps({'last_successful_page': page_num - 1, 'total_pages': total_pages, 'error_message': str(e)})}\n\n"
                    return

            try:
                (job_dir / "findings.json").write_text(
                    json.dumps(all_findings, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Could not save findings: {str(e)}'})}\n\n"

            yield f"event: done\ndata: {json.dumps({'total_findings': len(all_findings)})}\n\n"

        pdf_document.close()
        del pdf_document

        # ── Document-level check (only if document-scope checkpoints selected) ─
        if doc_checkpoints or job.get("doc_prompt"):
            yield f"event: document_start\ndata: {json.dumps({})}\n\n"
            # Re-fetch PDF bytes — freed early above to save memory during the page loop.
            try:
                _pdf_reload = await loop.run_in_executor(
                    None, partial(get_pdf_bytes_by_id, token, job["file_id"])
                )
                pdf_bytes = _pdf_reload.get("pdf_bytes")
                if not pdf_bytes:
                    raise ValueError("No PDF bytes retrieved")
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Could not reload PDF for document check: {str(e)}'})}\n\n"
                return
            try:
                doc_findings = await loop.run_in_executor(
                    None, partial(run_document_check, pdf_bytes, doc_checkpoints,
                                  job.get("doc_prompt") or None)
                )

                for finding in doc_findings:
                    finding["id"] = finding_id_counter
                    finding.setdefault("location", "Document")
                    all_findings.append(finding)
                    finding_id_counter += 1

                yield f"event: document_findings\ndata: {json.dumps({'findings': doc_findings})}\n\n"

                if doc_findings:
                    doc_reviews = await loop.run_in_executor(
                        None, partial(run_document_review, pdf_bytes, doc_findings)
                    )
                    yield f"event: document_review\ndata: {json.dumps({'reviews': doc_reviews})}\n\n"

                    doc_review_map = {r["finding_id"]: r for r in doc_reviews}
                    for f in doc_findings:
                        rev = doc_review_map.get(f["id"])
                        if rev:
                            f["review_status"] = rev["verdict"]
                            f["review_comment"] = rev["reason"]

                (job_dir / "findings.json").write_text(
                    json.dumps(all_findings, ensure_ascii=False), encoding="utf-8"
                )

            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Document-level check failed: {str(e)}'})}\n\n"

            del pdf_bytes  # free immediately after use

        asyncio.create_task(save_run_to_history(
            job_id=job_id,
            job=dict(job),
            all_findings=list(all_findings),
            total_pages=total_pages,
            token=token,
            job_dir=job_dir,
        ))

        yield f"event: all_done\ndata: {json.dumps({'total_findings': len(all_findings)})}\n\n"

        job["status"] = "completed"
        job.pop("last_successful_page", None)
        save_job(job_id, job)

    finally:
        state._ACTIVE_JOBS.discard(job_id)


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/check", response_class=HTMLResponse)
async def run_check(
    request: Request,
    drive_url: Annotated[str, Form()],
    workflow_id: Annotated[str, Form()],
    checkpoint_ids: Annotated[list[str], Form()] = [],
):
    """Validate input and create a job, then redirect to the processing page."""
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

    loop = asyncio.get_running_loop()
    try:
        file_data = await loop.run_in_executor(
            None, partial(get_file_as_pdf, token, drive_url.strip())
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

    job_id = uuid.uuid4().hex
    save_job(job_id, {
        "drive_url": drive_url.strip(),
        "checked_by": user["email"],
        "file_id": file_data["file_id"],
        "file_type": file_data["file_type"],
        "title": file_data["title"],
        "workflow_id": workflow_id,
        "checkpoint_ids": [cp["id"] for cp in selected_checkpoints],
        "status": "processing",
    })

    return RedirectResponse(url=f"/process/{job_id}", status_code=303)


@router.get("/process/{job_id}", response_class=HTMLResponse)
async def show_process(request: Request, job_id: str, retry_from: int = None):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    job = load_job(job_id)
    if not job:
        return RedirectResponse(url="/?error=Job+not+found.+Please+run+a+new+check.")

    checkpoint_names = {cp["id"]: cp["category"] for cp in state.CHECKPOINTS}

    return templates.TemplateResponse("process.html", ctx(
        request, user,
        job_id=job_id,
        title=job["title"],
        retry_from=retry_from,
        checkpoint_names=checkpoint_names,
    ))


@router.get("/stream/{job_id}")
async def stream_processing(request: Request, job_id: str, retry_from: int = None):
    """SSE endpoint for streaming page processing."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)

    if not user or not token:
        return RedirectResponse(url="/login")

    if job_id in state._ACTIVE_JOBS:
        async def _already_processing():
            yield (
                f"event: error\ndata: {json.dumps({'message': 'This job is already being processed. '
                'Please wait for the current run to complete.'})}\n\n"
            )
        return StreamingResponse(
            _already_processing(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        _stream_processing(job_id, token, retry_from),
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

    job = load_job(job_id)
    if not job:
        return RedirectResponse(url="/?error=Job+not+found.", status_code=303)

    from_page = job.get("last_successful_page", 0) + 1
    return RedirectResponse(url=f"/process/{job_id}?retry_from={from_page}", status_code=303)


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

    job = load_job(job_id)
    if not job:
        return {"error": "Job not found"}

    job_dir = state._JOBS_DIR / job_id
    findings_file = job_dir / "findings.json"
    if not findings_file.exists():
        return {"error": "Findings not found"}

    all_findings = json.loads(findings_file.read_text(encoding="utf-8"))
    selected_finding_ids = [int(fid) for fid in finding_ids if fid.isdigit()]
    selected_findings = [f for f in all_findings if f.get("id") in selected_finding_ids]

    file_data = {
        "file_id": job["file_id"],
        "file_type": job["file_type"],
        "title": job["title"],
    }

    loop = asyncio.get_running_loop()
    try:
        posted = await loop.run_in_executor(
            None, partial(post_selected_comments, token, file_data, selected_findings, state.CHECKPOINT_MAP)
        )
        return {"posted": posted, "total_selected": len(selected_findings)}
    except Exception as exc:
        return {"error": f"Could not post comments: {str(exc)}"}
