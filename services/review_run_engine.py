"""Review-run processing engine.

Runs are created and resumed here, and processed by a background asyncio task
that is independent of any SSE connection — closing the browser tab that
started a run does not stop it, and a redeploy or crash only pauses it (the
run is resumable from Supabase, not from anything on local disk).

All progress (status, last_successful_page, findings, page images) is written
to Supabase as it happens, not batched up for a single write at the end. That
is what makes both "resume from where it stopped" and "show up in History
while still running or after dying mid-way" possible.
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from functools import partial
from io import BytesIO

import fitz

import db
import state
from services.drive_service import (
    get_file_as_pdf, create_drive_subfolder, upload_jpeg_to_drive,
)
from services.review_ai import (
    run_vision_check, run_vision_review, run_document_check, run_document_review,
    _build_vision_prompt, _build_document_prompt,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Run creation / resume ───────────────────────────────────────────────────

async def create_run(
    token: dict,
    checked_by: str,
    drive_url: str,
    workflow: dict,
    selected_checkpoints: list[dict],
    custom_page_prompt: str | None = None,
    custom_doc_prompt: str | None = None,
) -> dict:
    """Fetch the Drive file, build prompts, insert the run row (status=processing),
    and start background processing. Returns {"job_id", "title"}.

    Raises ValueError for a bad/unreadable Drive link — callers translate that
    into their own error response (HTML template error vs. HTTPException).
    """
    loop = asyncio.get_running_loop()
    file_data = await loop.run_in_executor(None, partial(get_file_as_pdf, token, drive_url))

    page_cps = [cp for cp in selected_checkpoints if cp.get("scope") != "document"]
    doc_cps = [cp for cp in selected_checkpoints if cp.get("scope") == "document"]
    page_prompt = custom_page_prompt or _build_vision_prompt(page_cps, "{page_num}", workflow.get("name", ""))
    doc_prompt = custom_doc_prompt or (_build_document_prompt(doc_cps) if doc_cps else "")

    job_id = uuid.uuid4().hex
    run_row = {
        "id": job_id,
        "workflow_id": workflow.get("id", ""),
        "workflow_name": workflow.get("name", ""),
        "checked_by": checked_by,
        "document_name": file_data["title"],
        "drive_url": drive_url,
        "file_type": file_data["file_type"],
        "drive_folder_id": None,
        "checkpoint_ids": [cp["id"] for cp in selected_checkpoints],
        "total_pages": 0,
        "total_findings": 0,
        "valid_findings": 0,
        "invalid_findings": 0,
        "page_prompt": page_prompt,
        "doc_prompt": doc_prompt or None,
        "status": "processing",
        "last_successful_page": None,
        "doc_check_done": False,
        "error_message": None,
        "updated_at": _now_iso(),
    }
    await loop.run_in_executor(None, partial(db.insert_run, run_row))
    start_run_task(job_id, token)
    return {"job_id": job_id, "title": file_data["title"]}


async def resume_run(job_id: str, token: dict) -> dict:
    """Restart background processing for a run from its last saved page.

    Safe to call even if the run is stuck in status="processing" because the
    process that was running it died (server restart/crash) — that state is
    only trustworthy while `job_id` is also in state._ACTIVE_JOBS (this
    process's live-task registry); if it isn't, there is nothing actually
    running and it's safe (and necessary) to start a fresh task.

    Raises LookupError if the run doesn't exist, ValueError if it already
    completed (nothing to resume).
    """
    loop = asyncio.get_running_loop()
    run = await loop.run_in_executor(None, partial(db.fetch_run, job_id))
    if not run:
        raise LookupError("Run not found")
    if run.get("status") == "completed":
        raise ValueError("Run already completed")

    retry_from = (run.get("last_successful_page") or 0) + 1
    if job_id not in state._ACTIVE_JOBS:
        start_run_task(job_id, token)
    return {"job_id": job_id, "retry_from": retry_from}


def start_run_task(job_id: str, token: dict) -> None:
    if job_id in state._ACTIVE_JOBS:
        return
    state._ACTIVE_JOBS.add(job_id)
    asyncio.create_task(process_run(job_id, token))


# ── Findings persistence ─────────────────────────────────────────────────────

def _finding_row(run_id: str, f: dict) -> dict:
    return {
        "run_id": run_id,
        "page_num": f.get("page_num"),
        "checkpoint_id": f.get("checkpoint_id"),
        "quote": f.get("quote"),
        "location": f.get("location"),
        "issue": f.get("issue"),
        "suggestion": f.get("suggestion"),
        "review_status": None,
        "review_comment": None,
    }


async def _insert_findings(
    loop, job_id: str, page_num: int | None, findings: list[dict], is_document: bool,
) -> list[dict]:
    """Insert findings and annotate them with their real Supabase id — done as
    its own step (published to viewers before the review pass) so the live UI
    still shows the original "flagged, pending review" -> "valid/invalid" flow
    instead of findings only ever appearing already-reviewed."""
    if not findings:
        return []
    for f in findings:
        f["page_num"] = page_num
        if not f.get("location"):
            f["location"] = "Document" if is_document else f"Page {page_num}"
    rows = [_finding_row(job_id, f) for f in findings]
    inserted = await loop.run_in_executor(None, partial(db.insert_run_findings, rows))
    for f, row in zip(findings, inserted):
        f["id"] = row["id"]
    return findings


async def _review_findings(
    loop, page_num: int | None, findings: list[dict], media_bytes: bytes, is_document: bool,
) -> list[dict]:
    """Run the AI review pass on already-inserted findings, persist each verdict,
    and return the {finding_id, verdict, reason} rows (the shape the frontend's
    page_review/document_review handlers expect)."""
    if not findings:
        return []
    if is_document:
        reviews = await loop.run_in_executor(None, partial(run_document_review, media_bytes, findings))
    else:
        reviews = await loop.run_in_executor(None, partial(run_vision_review, media_bytes, findings, page_num))

    review_map = {r["finding_id"]: r for r in reviews}
    applied = []
    for f in findings:
        rev = review_map.get(f["id"])
        if rev:
            f["review_status"] = rev["verdict"]
            f["review_comment"] = rev["reason"]
            await loop.run_in_executor(
                None, partial(db.update_finding_review, str(f["id"]), rev["verdict"], rev["reason"])
            )
            applied.append(rev)
    return applied


# ── Background worker ────────────────────────────────────────────────────────

async def process_run(job_id: str, token: dict) -> None:
    """Process (or resume) a review run to completion, persisting progress to
    Supabase as it goes. Always ends with status "completed" or "failed" —
    never leaves a run silently stuck in "processing" from this process's
    point of view (a killed process is the one case this can't protect against,
    which is why History treats a long-stale "processing" run as resumable)."""
    loop = asyncio.get_running_loop()
    try:
        run = await loop.run_in_executor(None, partial(db.fetch_run, job_id))
        if not run:
            return

        checkpoint_ids = run.get("checkpoint_ids") or []
        selected_checkpoints = [
            state.CHECKPOINT_MAP[cid] for cid in checkpoint_ids if cid in state.CHECKPOINT_MAP
        ]
        page_checkpoints = [cp for cp in selected_checkpoints if cp.get("scope") != "document"]
        doc_checkpoints = [cp for cp in selected_checkpoints if cp.get("scope") == "document"]
        workflow_name = run.get("workflow_name") or run.get("workflow_id", "")

        try:
            file_data = await loop.run_in_executor(None, partial(get_file_as_pdf, token, run["drive_url"]))
            pdf_bytes = file_data["pdf_bytes"]
        except Exception as e:
            await loop.run_in_executor(None, partial(db.update_run, job_id, {
                "status": "failed", "error_message": f"Could not read the file: {e}", "updated_at": _now_iso(),
            }))
            state.publish(job_id, "error", {"message": f"Could not read the file: {e}"})
            return

        try:
            pdf_document = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
        except Exception as e:
            await loop.run_in_executor(None, partial(db.update_run, job_id, {
                "status": "failed", "error_message": f"Could not parse PDF: {e}", "updated_at": _now_iso(),
            }))
            state.publish(job_id, "error", {"message": f"Could not parse PDF: {e}"})
            return

        total_pages = len(pdf_document)
        if run.get("total_pages") != total_pages:
            await loop.run_in_executor(None, partial(db.update_run, job_id, {
                "total_pages": total_pages, "updated_at": _now_iso(),
            }))
        state.publish(job_id, "start", {"total_pages": total_pages, "title": run.get("document_name", "")})

        del pdf_bytes  # fitz has its own copy; free ours early like the old code did

        drive_folder_id = run.get("drive_folder_id")
        runs_folder_id = os.getenv("DRIVE_RUNS_FOLDER_ID")
        if not drive_folder_id and runs_folder_id:
            try:
                drive_folder_id = await loop.run_in_executor(
                    None, partial(create_drive_subfolder, token, runs_folder_id, job_id)
                )
                await loop.run_in_executor(None, partial(db.update_run, job_id, {
                    "drive_folder_id": drive_folder_id, "updated_at": _now_iso(),
                }))
            except Exception as e:
                print(f"[review] Drive folder creation failed for {job_id}: {e}")

        start_page = (run.get("last_successful_page") or 0) + 1

        if page_checkpoints and start_page <= total_pages:
            for page_num in range(start_page, total_pages + 1):
                try:
                    if page_num == start_page and run.get("last_successful_page"):
                        # This is a resume: page `start_page` is the one page that
                        # could have partial data from the attempt that failed
                        # (findings inserted but review/upload not finished) —
                        # clear it so reprocessing can't create duplicates.
                        await loop.run_in_executor(None, partial(db.delete_run_findings_for_slot, job_id, page_num))
                        await loop.run_in_executor(None, partial(db.delete_run_page, job_id, page_num))

                    page = pdf_document[page_num - 1]
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img_bytes = pix.tobytes(output="jpeg")

                    upload_future = None
                    if drive_folder_id:
                        upload_future = loop.run_in_executor(
                            None, partial(upload_jpeg_to_drive, token, drive_folder_id,
                                          f"page_{page_num:03d}.jpg", img_bytes)
                        )
                    ai_future = loop.run_in_executor(
                        None, partial(run_vision_check, img_bytes, page_checkpoints, page_num,
                                      workflow_name, run.get("page_prompt") or None)
                    )

                    drive_file_id = None
                    if upload_future:
                        try:
                            drive_file_id = await upload_future
                            await loop.run_in_executor(None, partial(db.insert_run_pages, [
                                {"run_id": job_id, "page_num": page_num, "drive_file_id": drive_file_id}
                            ]))
                        except Exception as e:
                            print(f"[review] Drive upload failed p{page_num}: {e}")

                    state.publish(job_id, "page_ready", {
                        "page": page_num, "total_pages": total_pages, "drive_file_id": drive_file_id,
                    })

                    findings = await ai_future
                    findings = await _insert_findings(loop, job_id, page_num, findings, is_document=False)
                    state.publish(job_id, "page_findings", {"page": page_num, "findings": findings})

                    if findings:
                        reviews = await _review_findings(loop, page_num, findings, img_bytes, is_document=False)
                        state.publish(job_id, "page_review", {"page": page_num, "reviews": reviews})

                    await loop.run_in_executor(None, partial(db.update_run, job_id, {
                        "last_successful_page": page_num, "updated_at": _now_iso(),
                    }))

                except Exception as e:
                    await loop.run_in_executor(None, partial(db.update_run, job_id, {
                        "status": "failed", "error_message": str(e), "updated_at": _now_iso(),
                    }))
                    state.publish(job_id, "partial_complete", {
                        "last_successful_page": page_num - 1,
                        "total_pages": total_pages,
                        "error_message": str(e),
                    })
                    return
            state.publish(job_id, "done", {})

        pdf_document.close()
        del pdf_document

        doc_check_needed = bool(doc_checkpoints) or bool(run.get("doc_prompt"))
        if doc_check_needed and not run.get("doc_check_done"):
            state.publish(job_id, "document_start", {})
            try:
                # Clears any findings left behind by a previous attempt that
                # inserted them but failed before finishing review (no-op if
                # this is the first attempt — nothing to delete).
                await loop.run_in_executor(None, partial(db.delete_run_findings_for_slot, job_id, None))
                doc_file_data = await loop.run_in_executor(
                    None, partial(get_file_as_pdf, token, run["drive_url"])
                )
                doc_pdf_bytes = doc_file_data["pdf_bytes"]
                doc_findings = await loop.run_in_executor(
                    None, partial(run_document_check, doc_pdf_bytes, doc_checkpoints, run.get("doc_prompt") or None)
                )
                doc_findings = await _insert_findings(loop, job_id, None, doc_findings, is_document=True)
                state.publish(job_id, "document_findings", {"findings": doc_findings})

                if doc_findings:
                    doc_reviews = await _review_findings(loop, None, doc_findings, doc_pdf_bytes, is_document=True)
                    state.publish(job_id, "document_review", {"reviews": doc_reviews})

                await loop.run_in_executor(None, partial(db.update_run, job_id, {
                    "doc_check_done": True, "updated_at": _now_iso(),
                }))
                del doc_pdf_bytes
            except Exception as e:
                await loop.run_in_executor(None, partial(db.update_run, job_id, {
                    "status": "failed", "error_message": f"Document-level check failed: {e}", "updated_at": _now_iso(),
                }))
                state.publish(job_id, "error", {"message": f"Document-level check failed: {e}"})
                return

        all_findings = await loop.run_in_executor(None, partial(db.fetch_run_findings, job_id))
        total_findings = len(all_findings)
        valid = sum(1 for f in all_findings if f.get("review_status") == "valid")
        invalid = sum(1 for f in all_findings if f.get("review_status") == "invalid")
        await loop.run_in_executor(None, partial(db.update_run, job_id, {
            "status": "completed",
            "total_findings": total_findings,
            "valid_findings": valid,
            "invalid_findings": invalid,
            "updated_at": _now_iso(),
        }))
        state.publish(job_id, "all_done", {"total_findings": total_findings})

    except Exception as e:
        try:
            await loop.run_in_executor(None, partial(db.update_run, job_id, {
                "status": "failed", "error_message": str(e), "updated_at": _now_iso(),
            }))
        except Exception:
            pass
        state.publish(job_id, "error", {"message": str(e)})
    finally:
        state._ACTIVE_JOBS.discard(job_id)
