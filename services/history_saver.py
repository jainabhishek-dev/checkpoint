"""Async helpers for persisting completed runs to Supabase and Google Drive."""

import asyncio
import os
from functools import partial
from pathlib import Path

import db
import state
from services.drive_service import create_drive_subfolder, upload_jpeg_to_drive


def apply_cic_verdict(current: str, new_verdict: str) -> str:
    """Merge a new AI verdict into the running verdict for a comment.

    Precedence: fixed > not_fixed > not_sure.
    Once fixed, always fixed. Once not_fixed (and not yet fixed), stays not_fixed.
    not_sure never overwrites a firmer verdict.
    """
    if current == "fixed":
        return "fixed"
    if new_verdict == "fixed":
        return "fixed"
    if current == "not_fixed":
        return "not_fixed"
    if new_verdict == "not_fixed":
        return "not_fixed"
    return "not_sure"


async def save_run_to_history(
    job_id: str,
    job: dict,
    all_findings: list,
    total_pages: int,
    token: dict,
    job_dir: Path,
) -> None:
    """
    Persist a completed review run to Supabase (immediately) then Google Drive
    (page images, slow). Runs as an independent asyncio.Task so that a client
    disconnect cannot cancel it.

    Order:
      1. insert_run (drive_folder_id=None) — run visible in history immediately
      2. insert_run_findings — findings visible immediately
      3. Upload page images to Drive (slow)
      4. insert_run_pages
      5. update_run to back-fill drive_folder_id
    """
    loop = asyncio.get_running_loop()
    runs_folder_id = os.getenv("DRIVE_RUNS_FOLDER_ID")
    wf = next((w for w in state.WORKFLOWS if w["id"] == job.get("workflow_id")), {})

    # ── Steps 1 & 2: Insert to Supabase immediately ───────────────────────────
    try:
        db.insert_run({
            "id": job_id,
            "workflow_id": job.get("workflow_id", ""),
            "workflow_name": wf.get("name", job.get("workflow_id", "")),
            "checked_by": job.get("checked_by", ""),
            "document_name": job.get("title"),
            "drive_url": job.get("drive_url"),
            "file_type": job.get("file_type"),
            "drive_folder_id": None,
            "checkpoint_ids": job.get("checkpoint_ids", []),
            "total_pages": total_pages,
            "total_findings": len(all_findings),
            "valid_findings": sum(1 for f in all_findings if f.get("review_status") == "valid"),
            "invalid_findings": sum(1 for f in all_findings if f.get("review_status") == "invalid"),
            "page_prompt": job.get("page_prompt"),
            "doc_prompt": job.get("doc_prompt") or None,
        })
        if all_findings:
            db.insert_run_findings([
                {
                    "run_id": job_id,
                    "page_num": f.get("page_num"),
                    "checkpoint_id": f.get("checkpoint_id"),
                    "quote": f.get("quote"),
                    "location": f.get("location"),
                    "issue": f.get("issue"),
                    "suggestion": f.get("suggestion"),
                    "review_status": f.get("review_status"),
                    "review_comment": f.get("review_comment"),
                }
                for f in all_findings
            ])
        print(f"[history] Run {job_id} saved to Supabase: {total_pages} pages, {len(all_findings)} findings.")
    except Exception as e:
        print(f"[history] Supabase save failed for {job_id}: {e}")
        return

    # ── Steps 3–5: Upload images to Drive then back-fill folder ID ────────────
    if not runs_folder_id:
        return

    page_records: list[dict] = []
    try:
        drive_folder_id = await loop.run_in_executor(
            None, partial(create_drive_subfolder, token, runs_folder_id, job_id)
        )
        for img_path in sorted(job_dir.glob("page_*.jpg")):
            pg = int(img_path.stem.split("_")[1])
            img_data = img_path.read_bytes()
            file_id = await loop.run_in_executor(
                None, partial(upload_jpeg_to_drive, token, drive_folder_id, img_path.name, img_data)
            )
            page_records.append({"run_id": job_id, "page_num": pg, "drive_file_id": file_id})
            img_path.unlink(missing_ok=True)  # delete local file after successful Drive upload

        if page_records:
            db.insert_run_pages(page_records)
        db.update_run(job_id, {"drive_folder_id": drive_folder_id})
        print(f"[history] Run {job_id} Drive upload complete: {len(page_records)} images.")
    except Exception as e:
        print(f"[history] Drive upload failed for {job_id}: {e}")


async def save_cic_run_to_history(
    job_id: str,
    job: dict,
    comment_tracker: dict,
    total_pages: int,
    token: dict,
    job_dir: Path,
) -> None:
    """
    Persist a completed CIC run to Supabase (immediately) then Google Drive
    (page images, slow). Runs as an independent asyncio.Task so client disconnect
    cannot cancel it.

    Order:
      1. insert_cic_run (drive_folder_id=None) — visible in history immediately
      2. insert_cic_comments — verdicts visible immediately
      3. Upload page images to Drive (slow)
      4. insert_cic_run_pages
      5. update_cic_run to back-fill drive_folder_id
    """
    loop = asyncio.get_running_loop()
    runs_folder_id = os.getenv("DRIVE_RUNS_FOLDER_ID")

    fixed = sum(1 for c in comment_tracker.values() if c["verdict"] == "fixed")
    not_fixed = sum(1 for c in comment_tracker.values() if c["verdict"] == "not_fixed")
    not_sure = sum(1 for c in comment_tracker.values() if c["verdict"] == "not_sure")

    # ── Steps 1 & 2: Insert to Supabase immediately ───────────────────────────
    try:
        db.insert_cic_run({
            "id": job_id,
            "workflow_id": job.get("workflow_id", ""),
            "workflow_name": job.get("workflow_name", ""),
            "checked_by": job.get("checked_by", ""),
            "commented_file_name": job.get("commented_file_title"),
            "commented_drive_url": job.get("commented_drive_url"),
            "revised_file_name": job.get("revised_file_title"),
            "revised_drive_url": job.get("revised_drive_url"),
            "drive_folder_id": None,
            "total_pages": total_pages,
            "total_comments": len(comment_tracker),
            "fixed_count": fixed,
            "not_fixed_count": not_fixed,
            "not_sure_count": not_sure,
        })
        if comment_tracker:
            comment_rows = [
                {
                    "run_id": job_id,
                    "comment_id": cid,
                    "author": info.get("author", ""),
                    "content": info.get("content", ""),
                    "verdict": info["verdict"],
                    "reason": info.get("reason", ""),
                    "page_resolved": info.get("page_resolved"),
                    "original_page": info.get("original_page"),
                }
                for cid, info in comment_tracker.items()
            ]
            db.insert_cic_comments(comment_rows)
        print(f"[cic-history] Run {job_id} saved to Supabase: {total_pages} pages, {len(comment_tracker)} comments.")
    except Exception as e:
        print(f"[cic-history] Supabase save failed for {job_id}: {e}")
        return

    # ── Steps 3–5: Upload images to Drive then back-fill folder ID ────────────
    if not runs_folder_id:
        return

    page_records: list[dict] = []
    try:
        drive_folder_id = await loop.run_in_executor(
            None, partial(create_drive_subfolder, token, runs_folder_id, f"cic_{job_id}")
        )
        for img_path in sorted(job_dir.glob("f?_page_*.jpg")):
            parts = img_path.stem.split("_")  # ["f1","page","001"] or ["f2","page","001"]
            file_version = "commented" if parts[0] == "f1" else "revised"
            pg = int(parts[2])
            img_data = img_path.read_bytes()
            drive_file_id = await loop.run_in_executor(
                None, partial(upload_jpeg_to_drive, token, drive_folder_id, img_path.name, img_data)
            )
            page_records.append({
                "run_id": job_id,
                "page_num": pg,
                "file_version": file_version,
                "drive_file_id": drive_file_id,
            })
            img_path.unlink(missing_ok=True)  # delete local file after successful Drive upload

        if page_records:
            db.insert_cic_run_pages(page_records)
        db.update_cic_run(job_id, {"drive_folder_id": drive_folder_id})
        print(f"[cic-history] Run {job_id} Drive upload complete: {len(page_records)} images.")
    except Exception as e:
        print(f"[cic-history] Drive upload failed for {job_id}: {e}")


async def save_ak_run_to_history(
    job_id: str,
    job: dict,
    question_results: list[dict],
) -> None:
    """
    Persist a completed AK Review run to Supabase immediately.
    No Drive image uploads — AK Review is table-only.

    Order:
      1. insert_ak_run — run visible in history immediately
      2. insert_ak_question_results — all question rows visible immediately
    """
    total = len(question_results)
    present = sum(1 for r in question_results if r.get("present_in_ak") == "Yes")
    missing = sum(1 for r in question_results if r.get("present_in_ak") == "No")
    incorrect = sum(1 for r in question_results if r.get("answer_correct") == "No")
    manual = sum(1 for r in question_results if r.get("answer_correct") == "Manual Review Required")

    try:
        db.insert_ak_run({
            "id": job_id,
            "workflow_id": job.get("workflow_id", ""),
            "workflow_name": job.get("workflow_name", ""),
            "checked_by": job.get("checked_by", ""),
            "chapter_file_name": job.get("chapter_file_title"),
            "chapter_drive_url": job.get("chapter_drive_url"),
            "ak_file_name": job.get("ak_file_title"),
            "ak_drive_url": job.get("ak_drive_url"),
            "prompt": job.get("prompt"),
            "total_questions": total,
            "present_in_ak": present,
            "missing_from_ak": missing,
            "incorrect_answers": incorrect,
            "manual_review_cases": manual,
        })
        if question_results:
            db.insert_ak_question_results([
                {
                    "run_id": job_id,
                    "page_no": r.get("page_no"),
                    "exercise_no": r.get("exercise_no"),
                    "question_no": r.get("question_no"),
                    "present_in_ak": r.get("present_in_ak"),
                    "answer_correct": r.get("answer_correct"),
                    "suggestions": r.get("suggestions"),
                }
                for r in question_results
            ])
        print(f"[ak-history] Run {job_id} saved to Supabase: {total} questions reviewed.")
    except Exception as e:
        print(f"[ak-history] Supabase save failed for {job_id}: {e}")
