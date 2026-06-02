"""CIC workflow routes: /cic-check, /cic-process/{id}, /cic-stream/{id},
/cic-job/{id}/{ver}/{page}.
"""

import asyncio
import json
import re
from functools import partial

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from starlette.requests import Request
from typing import Annotated

import uuid

import auth
import state
from utils import ctx, ensure_job_dir, load_job, save_job, templates
from services.drive_service import (
    get_file_as_pdf, get_pdf_bytes_by_id,
    fetch_drive_comments_with_pages, extract_pdf_annotations,
)
from services.cic_ai import run_cic_check, run_cic_global_check
from services.history_saver import apply_cic_verdict, save_cic_run_to_history

router = APIRouter()


# ── SSE streaming generator ────────────────────────────────────────────────────

async def _stream_cic_processing(job_id: str, token: dict):
    """SSE generator for CIC processing: compares two PDFs page-by-page."""
    import fitz
    from io import BytesIO

    job = load_job(job_id)
    if not job:
        yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
        return

    state._ACTIVE_JOBS.add(job_id)
    try:
        loop = asyncio.get_running_loop()
        job_dir = ensure_job_dir(job_id)

        all_comments: list[dict] = job.get("comments", [])
        if not all_comments:
            yield f"event: error\ndata: {json.dumps({'message': 'No comments found in job.'})}\n\n"
            return

        # Initialise verdict tracker
        comment_tracker: dict[str, dict] = {}
        for c in all_comments:
            comment_tracker[c["id"]] = {
                "content": c.get("content", ""),
                "author": c.get("author", ""),
                "verdict": "not_sure",
                "reason": "",
                "page_resolved": None,
                "original_page": None,
            }

        # Load both PDFs
        try:
            f1_data = await loop.run_in_executor(
                None, partial(get_pdf_bytes_by_id, token, job["commented_file_id"])
            )
            f1_bytes = f1_data["pdf_bytes"]
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': f'Could not load commented file: {str(e)}'})}\n\n"
            return

        try:
            f2_data = await loop.run_in_executor(
                None, partial(get_pdf_bytes_by_id, token, job["revised_file_id"])
            )
            f2_bytes = f2_data["pdf_bytes"]
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': f'Could not load revised file: {str(e)}'})}\n\n"
            return

        # Enrich page_num for Drive comments that lack an anchor (e.g. Adobe PDF
        # annotations). Comments added natively in Drive already have page_num set —
        # those are left untouched. Only comments with page_num = None are matched.
        try:
            fitz_annots = await loop.run_in_executor(
                None, partial(extract_pdf_annotations, f1_bytes)
            )
            if fitz_annots:
                def _norm(s: str) -> str:
                    return re.sub(r'\s+', ' ', (s or '').strip())

                unmatched = list(fitz_annots)
                for c in all_comments:
                    if c.get("page_num") is not None:
                        continue
                    key = _norm(c.get("content", ""))
                    if not key:
                        continue
                    for i, annot in enumerate(unmatched):
                        if _norm(annot["content"]) == key:
                            c["page_num"] = annot["page_num"]
                            unmatched.pop(i)
                            break
        except Exception as e:
            print(f"[cic] fitz annotation extraction failed for {job_id}: {e}")

        # Split comments into page buckets and global bucket (after enrichment)
        page_comments_map: dict[int, list[dict]] = {}
        global_comments: list[dict] = []
        for c in all_comments:
            pn = c.get("page_num")
            if pn and isinstance(pn, int) and pn > 0:
                page_comments_map.setdefault(pn, []).append(c)
            else:
                global_comments.append(c)

        # Record original_page for every comment that was mapped to a specific page
        for pn, pcomments in page_comments_map.items():
            for c in pcomments:
                comment_tracker[c["id"]]["original_page"] = pn

        try:
            f1_doc = fitz.open(stream=BytesIO(f1_bytes), filetype="pdf")
            f2_doc = fitz.open(stream=BytesIO(f2_bytes), filetype="pdf")
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': f'Could not parse PDFs: {str(e)}'})}\n\n"
            return

        # Free PDF bytes early — fitz has loaded them into its own C-level memory.
        # They will be re-fetched below only if the global check is needed.
        del f1_bytes, f2_bytes

        total_pages = max(len(f1_doc), len(f2_doc))

        yield f"event: cic_start\ndata: {json.dumps({'total_comments': len(all_comments), 'total_pages': total_pages})}\n\n"

        # Page-by-page processing
        for page_num in range(1, total_pages + 1):
            try:
                mat = fitz.Matrix(2, 2)

                if page_num <= len(f1_doc):
                    pix1 = f1_doc[page_num - 1].get_pixmap(matrix=mat, alpha=False)
                    f1_img = pix1.tobytes(output="jpeg")
                else:
                    f1_img = None

                if page_num <= len(f2_doc):
                    pix2 = f2_doc[page_num - 1].get_pixmap(matrix=mat, alpha=False)
                    f2_img = pix2.tobytes(output="jpeg")
                else:
                    f2_img = None

                if f1_img:
                    (job_dir / f"f1_page_{page_num:03d}.jpg").write_bytes(f1_img)
                if f2_img:
                    (job_dir / f"f2_page_{page_num:03d}.jpg").write_bytes(f2_img)

                page_comments = page_comments_map.get(page_num, [])
                page_verdicts = []

                if page_comments and f1_img and f2_img:
                    verdicts = await loop.run_in_executor(
                        None, partial(run_cic_check, f1_img, f2_img, page_num, page_comments)
                    )
                    verdict_map = {v["comment_id"]: v for v in verdicts}
                    for c in page_comments:
                        cid = c["id"]
                        ai_result = verdict_map.get(cid)
                        if ai_result:
                            old_verdict = comment_tracker[cid]["verdict"]
                            new_v = ai_result["verdict"]
                            merged = apply_cic_verdict(old_verdict, new_v)
                            if merged != old_verdict:
                                comment_tracker[cid]["verdict"] = merged
                                comment_tracker[cid]["reason"] = ai_result["reason"]
                                comment_tracker[cid]["page_resolved"] = page_num
                            page_verdicts.append({
                                "comment_id": cid,
                                "content": c.get("content", ""),
                                "verdict": merged,
                                "reason": ai_result["reason"],
                            })

                del f1_img, f2_img

                yield f"event: cic_page\ndata: {json.dumps({'page_num': page_num, 'total_pages': total_pages, 'verdicts': page_verdicts, 'comment_count': len(page_comments)})}\n\n"

            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Error on page {page_num}: {str(e)}'})}\n\n"
                return

        f1_doc.close()
        f2_doc.close()

        # Global pass for any still-not_sure comments
        not_sure_comments = [
            {"id": cid, "content": info["content"], "author": info["author"]}
            for cid, info in comment_tracker.items()
            if info["verdict"] == "not_sure"
        ]

        global_verdict_results = []
        if not_sure_comments:
            yield f"event: cic_global_start\ndata: {json.dumps({'count': len(not_sure_comments)})}\n\n"
            # Re-fetch both PDFs — they were freed early above to save memory during the page loop.
            try:
                _f1 = await loop.run_in_executor(
                    None, partial(get_pdf_bytes_by_id, token, job["commented_file_id"])
                )
                _f2 = await loop.run_in_executor(
                    None, partial(get_pdf_bytes_by_id, token, job["revised_file_id"])
                )
                _f1_bytes = _f1["pdf_bytes"]
                _f2_bytes = _f2["pdf_bytes"]
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Could not reload files for global check: {str(e)}'})}\n\n"
                return
            try:
                global_verdicts = await loop.run_in_executor(
                    None, partial(run_cic_global_check, _f1_bytes, _f2_bytes, not_sure_comments)
                )
                del _f1_bytes, _f2_bytes  # free immediately after use
                verdict_map = {v["comment_id"]: v for v in global_verdicts}
                for c in not_sure_comments:
                    cid = c["id"]
                    ai_result = verdict_map.get(cid)
                    if ai_result:
                        comment_tracker[cid]["verdict"] = ai_result["verdict"]
                        comment_tracker[cid]["reason"] = ai_result["reason"]
                        global_verdict_results.append({
                            "comment_id": cid,
                            "content": c["content"],
                            "verdict": ai_result["verdict"],
                            "reason": ai_result["reason"],
                            "original_page": comment_tracker[cid]["original_page"],
                        })
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Global check failed: {str(e)}'})}\n\n"
                return

            yield f"event: cic_global\ndata: {json.dumps({'verdicts': global_verdict_results})}\n\n"

        fixed_count = sum(1 for c in comment_tracker.values() if c["verdict"] == "fixed")
        not_fixed_count = sum(1 for c in comment_tracker.values() if c["verdict"] == "not_fixed")
        not_sure_count = sum(1 for c in comment_tracker.values() if c["verdict"] == "not_sure")

        asyncio.create_task(save_cic_run_to_history(
            job_id=job_id,
            job=dict(job),
            comment_tracker={k: dict(v) for k, v in comment_tracker.items()},
            total_pages=total_pages,
            token=token,
            job_dir=job_dir,
        ))

        yield f"event: cic_done\ndata: {json.dumps({'run_id': job_id, 'total_comments': len(all_comments), 'fixed': fixed_count, 'not_fixed': not_fixed_count, 'not_sure': not_sure_count})}\n\n"

        job["status"] = "completed"
        save_job(job_id, job)

    finally:
        state._ACTIVE_JOBS.discard(job_id)


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/cic-check", response_class=HTMLResponse)
async def run_cic_check_route(
    request: Request,
    workflow_id: Annotated[str, Form()],
    commented_url: Annotated[str, Form()],
    revised_url: Annotated[str, Form()],
):
    """Validate CIC inputs, fetch comments, create job, redirect to processing page."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)

    if not user or not token:
        return RedirectResponse(url="/login", status_code=303)

    workflow = next((w for w in state.WORKFLOWS if w["id"] == workflow_id), None)
    if not workflow or workflow.get("type") != "cic":
        return RedirectResponse(url="/?error=Invalid+CIC+workflow.", status_code=303)

    commented_url = commented_url.strip()
    revised_url = revised_url.strip()
    if not commented_url or not revised_url:
        return RedirectResponse(url=f"/?workflow={workflow_id}&error=Both+file+URLs+are+required.", status_code=303)

    loop = asyncio.get_running_loop()

    try:
        f1_data = await loop.run_in_executor(None, partial(get_file_as_pdf, token, commented_url))
    except Exception as exc:
        err = str(exc)
        if "invalid_grant" in err or "Token has been expired" in err:
            request.session.clear()
            return RedirectResponse(url="/login", status_code=303)
        return RedirectResponse(url=f"/?workflow={workflow_id}&error=Commented+file+error:+{err}", status_code=303)

    try:
        f2_data = await loop.run_in_executor(None, partial(get_file_as_pdf, token, revised_url))
    except Exception as exc:
        err = str(exc)
        if "invalid_grant" in err or "Token has been expired" in err:
            request.session.clear()
            return RedirectResponse(url="/login", status_code=303)
        return RedirectResponse(url=f"/?workflow={workflow_id}&error=Revised+file+error:+{err}", status_code=303)

    if f1_data["file_type"] != "pdf":
        return RedirectResponse(url=f"/?workflow={workflow_id}&error=Commented+file+must+be+a+PDF.", status_code=303)
    if f2_data["file_type"] != "pdf":
        return RedirectResponse(url=f"/?workflow={workflow_id}&error=Revised+file+must+be+a+PDF.", status_code=303)

    try:
        comments = await loop.run_in_executor(
            None, partial(fetch_drive_comments_with_pages, token, f1_data["file_id"])
        )
    except Exception as exc:
        err = str(exc)
        return RedirectResponse(url=f"/?workflow={workflow_id}&error=Could+not+fetch+comments:+{err}", status_code=303)

    if not comments:
        return RedirectResponse(
            url=f"/?workflow={workflow_id}&error=No+unresolved+comments+found+on+the+commented+file.",
            status_code=303,
        )

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

    return RedirectResponse(url=f"/cic-process/{job_id}", status_code=303)


@router.get("/cic-process/{job_id}", response_class=HTMLResponse)
async def show_cic_process(request: Request, job_id: str):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    job = load_job(job_id)
    if not job:
        return RedirectResponse(url="/?error=Job+not+found.+Please+run+a+new+check.")

    return templates.TemplateResponse("cic_process.html", ctx(
        request, user,
        job_id=job_id,
        commented_title=job.get("commented_file_title", ""),
        revised_title=job.get("revised_file_title", ""),
        total_comments=len(job.get("comments", [])),
    ))


@router.get("/cic-stream/{job_id}")
async def stream_cic_processing(request: Request, job_id: str):
    """SSE endpoint for CIC job streaming."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)

    if not user or not token:
        return RedirectResponse(url="/login")

    if job_id in state._ACTIVE_JOBS:
        async def _already_running():
            yield f"event: error\ndata: {json.dumps({'message': 'This job is already being processed.'})}\n\n"
        return StreamingResponse(
            _already_running(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        _stream_cic_processing(job_id, token),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/cic-job/{job_id}/{file_version}/{page_num}")
async def serve_cic_page_image(job_id: str, file_version: str, page_num: int):
    """Serve a CIC page image from disk. file_version is 'f1' or 'f2'."""
    if file_version not in ("f1", "f2"):
        raise HTTPException(status_code=400, detail="file_version must be f1 or f2")
    job_dir = state._JOBS_DIR / job_id
    page_file = job_dir / f"{file_version}_page_{page_num:03d}.jpg"
    if not page_file.exists():
        raise HTTPException(status_code=404, detail="Page image not found")
    return FileResponse(page_file, media_type="image/jpeg")
