"""Supabase data-access helpers using httpx REST API directly."""

import os
import httpx

_HEADERS_CACHE: dict | None = None


def _headers() -> dict:
    global _HEADERS_CACHE
    if _HEADERS_CACHE is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
        _HEADERS_CACHE = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
    return _HEADERS_CACHE


def _base_url() -> str:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    return f"{url}/rest/v1"


def fetch_all_checkpoints() -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/checkpoints",
        headers=_headers(),
        params={"order": "sort_order"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def insert_checkpoint(row: dict) -> dict:
    resp = httpx.post(
        f"{_base_url()}/checkpoints",
        headers=_headers(),
        json=row,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) else data


def update_checkpoint(cp_id: str, fields: dict) -> dict:
    resp = httpx.patch(
        f"{_base_url()}/checkpoints",
        headers=_headers(),
        params={"id": f"eq.{cp_id}"},
        json=fields,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) else data


def delete_checkpoint(cp_id: str) -> None:
    resp = httpx.delete(
        f"{_base_url()}/checkpoints",
        headers=_headers(),
        params={"id": f"eq.{cp_id}"},
        timeout=10,
    )
    resp.raise_for_status()


# ── Workflow helpers ──────────────────────────────────────────────────────────

def fetch_all_workflows() -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/workflows",
        headers=_headers(),
        params={"order": "sort_order"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def insert_workflow(row: dict) -> dict:
    resp = httpx.post(
        f"{_base_url()}/workflows",
        headers=_headers(),
        json=row,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) else data


def update_workflow(wf_id: str, fields: dict) -> dict:
    resp = httpx.patch(
        f"{_base_url()}/workflows",
        headers=_headers(),
        params={"id": f"eq.{wf_id}"},
        json=fields,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) else data


def delete_workflow(wf_id: str) -> None:
    resp = httpx.delete(
        f"{_base_url()}/workflows",
        headers=_headers(),
        params={"id": f"eq.{wf_id}"},
        timeout=10,
    )
    resp.raise_for_status()


def fetch_checkpoints_by_workflow(wf_id: str) -> list[dict]:
    """Fetch all checkpoints whose workflows array contains wf_id."""
    resp = httpx.get(
        f"{_base_url()}/checkpoints",
        headers=_headers(),
        params={"workflows": f'cs.{{"{wf_id}"}}'},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ── Admin helpers ─────────────────────────────────────────────────────────────

def fetch_all_admins() -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/admins",
        headers=_headers(),
        params={"order": "added_at"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def insert_admin(row: dict) -> dict:
    resp = httpx.post(
        f"{_base_url()}/admins",
        headers=_headers(),
        json=row,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) else data


def delete_admin(email: str) -> None:
    resp = httpx.delete(
        f"{_base_url()}/admins",
        headers=_headers(),
        params={"email": f"eq.{email}"},
        timeout=10,
    )
    resp.raise_for_status()


# ── Run history helpers ───────────────────────────────────────────────────────

def insert_run(row: dict) -> None:
    resp = httpx.post(
        f"{_base_url()}/runs",
        headers=_headers(),
        json=row,
        timeout=15,
    )
    resp.raise_for_status()


def update_run(run_id: str, fields: dict) -> None:
    """PATCH arbitrary fields on a run row (status, progress, totals, etc.)."""
    resp = httpx.patch(
        f"{_base_url()}/runs",
        headers=_headers(),
        params={"id": f"eq.{run_id}"},
        json=fields,
        timeout=15,
    )
    resp.raise_for_status()


def insert_run_pages(rows: list[dict]) -> None:
    """Batch-insert all page image records for a run in one call."""
    resp = httpx.post(
        f"{_base_url()}/run_pages",
        headers=_headers(),
        json=rows,
        timeout=15,
    )
    resp.raise_for_status()


def delete_run_page(run_id: str, page_num: int) -> None:
    """Delete any existing page-image record for this page. Called before
    reprocessing a page on resume so a retry can't leave two Drive image
    records (one orphaned) for the same page_num."""
    resp = httpx.delete(
        f"{_base_url()}/run_pages",
        headers=_headers(),
        params={"run_id": f"eq.{run_id}", "page_num": f"eq.{page_num}"},
        timeout=10,
    )
    resp.raise_for_status()


def delete_run_findings_for_slot(run_id: str, page_num: int | None) -> None:
    """Delete any existing findings for a specific page (or the document-level
    slot when page_num is None). Called before reprocessing that page/step on
    resume — makes it idempotent instead of appending duplicate finding rows
    if a previous attempt inserted findings but failed before finishing review."""
    params = {"run_id": f"eq.{run_id}"}
    if page_num is None:
        params["page_num"] = "is.null"
    else:
        params["page_num"] = f"eq.{page_num}"
    resp = httpx.delete(
        f"{_base_url()}/run_findings",
        headers=_headers(),
        params=params,
        timeout=10,
    )
    resp.raise_for_status()


def insert_run_findings(rows: list[dict]) -> list[dict]:
    """Insert findings and return the inserted rows (including their generated ids),
    in the same order as `rows` — callers rely on this to correlate findings with
    their real Supabase id for the AI review pass and for later /findings/{id}/review calls."""
    resp = httpx.post(
        f"{_base_url()}/run_findings",
        headers=_headers(),
        json=rows,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_runs(workflow_id: str | None = None) -> list[dict]:
    """Fetch all runs ordered by most recent first, optionally filtered by workflow."""
    params: dict = {"order": "created_at.desc"}
    if workflow_id:
        params["workflow_id"] = f"eq.{workflow_id}"
    resp = httpx.get(
        f"{_base_url()}/runs",
        headers=_headers(),
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_total_count(content_range: str | None) -> int:
    """Parse PostgREST's `Content-Range: 0-19/97` response header into 97."""
    if not content_range or "/" not in content_range:
        return 0
    total_str = content_range.rsplit("/", 1)[-1]
    return int(total_str) if total_str.isdigit() else 0


def _fetch_page(table: str, workflow_id: str | None, limit: int, offset: int) -> tuple[list[dict], int]:
    """Fetch one page of rows from `table` (most recent first) plus the total
    row count, via PostgREST's native limit/offset + `Prefer: count=exact`."""
    params: dict = {"order": "created_at.desc", "limit": str(limit), "offset": str(offset)}
    if workflow_id:
        params["workflow_id"] = f"eq.{workflow_id}"
    headers = {**_headers(), "Prefer": "count=exact"}
    resp = httpx.get(f"{_base_url()}/{table}", headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json(), _parse_total_count(resp.headers.get("content-range"))


def fetch_runs_page(workflow_id: str | None, limit: int, offset: int) -> tuple[list[dict], int]:
    return _fetch_page("runs", workflow_id, limit, offset)


def fetch_cic_runs_page(workflow_id: str | None, limit: int, offset: int) -> tuple[list[dict], int]:
    return _fetch_page("cic_runs", workflow_id, limit, offset)


def fetch_ak_runs_page(workflow_id: str | None, limit: int, offset: int) -> tuple[list[dict], int]:
    return _fetch_page("ak_runs", workflow_id, limit, offset)




def fetch_run(run_id: str) -> dict | None:
    resp = httpx.get(
        f"{_base_url()}/runs",
        headers=_headers(),
        params={"id": f"eq.{run_id}", "limit": "1"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None


def fetch_run_pages(run_id: str) -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/run_pages",
        headers=_headers(),
        params={"run_id": f"eq.{run_id}", "order": "page_num"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_all_run_page_image_ids() -> list[str]:
    """Return all drive_file_id values from run_pages (for permission backfill)."""
    resp = httpx.get(
        f"{_base_url()}/run_pages",
        headers=_headers(),
        params={"select": "drive_file_id", "drive_file_id": "not.is.null"},
        timeout=30,
    )
    resp.raise_for_status()
    return [r["drive_file_id"] for r in resp.json() if r.get("drive_file_id")]


def fetch_all_cic_page_image_ids() -> list[str]:
    """Return all drive_file_id values from cic_run_pages (for permission backfill)."""
    resp = httpx.get(
        f"{_base_url()}/cic_run_pages",
        headers=_headers(),
        params={"select": "drive_file_id", "drive_file_id": "not.is.null"},
        timeout=30,
    )
    resp.raise_for_status()
    return [r["drive_file_id"] for r in resp.json() if r.get("drive_file_id")]


def fetch_run_findings(run_id: str) -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/run_findings",
        headers=_headers(),
        params={"run_id": f"eq.{run_id}", "order": "page_num.asc.nullslast,id.asc"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ── CIC run helpers ───────────────────────────────────────────────────────────

def insert_cic_run(row: dict) -> None:
    resp = httpx.post(
        f"{_base_url()}/cic_runs",
        headers=_headers(),
        json=row,
        timeout=15,
    )
    resp.raise_for_status()


def insert_cic_run_pages(rows: list[dict]) -> None:
    """Batch-insert all page image records for a CIC run."""
    resp = httpx.post(
        f"{_base_url()}/cic_run_pages",
        headers=_headers(),
        json=rows,
        timeout=15,
    )
    resp.raise_for_status()


def insert_cic_comments(rows: list[dict]) -> None:
    """Batch-insert all comment verdict records for a CIC run."""
    resp = httpx.post(
        f"{_base_url()}/cic_comments",
        headers=_headers(),
        json=rows,
        timeout=15,
    )
    resp.raise_for_status()


def fetch_cic_runs(workflow_id: str | None = None) -> list[dict]:
    """Fetch all CIC runs ordered by most recent first, optionally filtered by workflow."""
    params: dict = {"order": "created_at.desc"}
    if workflow_id:
        params["workflow_id"] = f"eq.{workflow_id}"
    resp = httpx.get(
        f"{_base_url()}/cic_runs",
        headers=_headers(),
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_cic_run(run_id: str) -> dict | None:
    resp = httpx.get(
        f"{_base_url()}/cic_runs",
        headers=_headers(),
        params={"id": f"eq.{run_id}", "limit": "1"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None


def fetch_cic_run_pages(run_id: str) -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/cic_run_pages",
        headers=_headers(),
        params={"run_id": f"eq.{run_id}", "order": "page_num,file_version"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_cic_comments(run_id: str) -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/cic_comments",
        headers=_headers(),
        params={"run_id": f"eq.{run_id}", "order": "page_resolved.asc.nullslast,id.asc"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def update_run(run_id: str, fields: dict) -> None:
    """PATCH arbitrary fields on a run row (e.g. back-fill drive_folder_id)."""
    resp = httpx.patch(
        f"{_base_url()}/runs",
        headers=_headers(),
        params={"id": f"eq.{run_id}"},
        json=fields,
        timeout=10,
    )
    resp.raise_for_status()


def update_cic_run(run_id: str, fields: dict) -> None:
    """PATCH arbitrary fields on a cic_run row (e.g. back-fill drive_folder_id)."""
    resp = httpx.patch(
        f"{_base_url()}/cic_runs",
        headers=_headers(),
        params={"id": f"eq.{run_id}"},
        json=fields,
        timeout=10,
    )
    resp.raise_for_status()


def update_finding_review(finding_id: str, review_status: str, review_comment: str) -> None:
    """Overwrite the review verdict and comment on a single finding."""
    resp = httpx.patch(
        f"{_base_url()}/run_findings",
        headers=_headers(),
        params={"id": f"eq.{finding_id}"},
        json={"review_status": review_status, "review_comment": review_comment},
        timeout=10,
    )
    resp.raise_for_status()


# ── AK Review run helpers ─────────────────────────────────────────────────────

def insert_ak_run(row: dict) -> None:
    resp = httpx.post(
        f"{_base_url()}/ak_runs",
        headers=_headers(),
        json=row,
        timeout=15,
    )
    resp.raise_for_status()


def insert_ak_question_results(rows: list[dict]) -> None:
    """Batch-insert all question result records for an AK run."""
    resp = httpx.post(
        f"{_base_url()}/ak_question_results",
        headers=_headers(),
        json=rows,
        timeout=30,
    )
    resp.raise_for_status()


def fetch_ak_runs(workflow_id: str | None = None) -> list[dict]:
    """Fetch all AK runs ordered by most recent first, optionally filtered by workflow."""
    params: dict = {"order": "created_at.desc"}
    if workflow_id:
        params["workflow_id"] = f"eq.{workflow_id}"
    resp = httpx.get(
        f"{_base_url()}/ak_runs",
        headers=_headers(),
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_ak_run(run_id: str) -> dict | None:
    resp = httpx.get(
        f"{_base_url()}/ak_runs",
        headers=_headers(),
        params={"id": f"eq.{run_id}", "limit": "1"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None


def fetch_ak_question_results(run_id: str) -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/ak_question_results",
        headers=_headers(),
        params={"run_id": f"eq.{run_id}", "order": "id.asc"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def update_ak_run(run_id: str, fields: dict) -> None:
    """PATCH arbitrary fields on an ak_run row."""
    resp = httpx.patch(
        f"{_base_url()}/ak_runs",
        headers=_headers(),
        params={"id": f"eq.{run_id}"},
        json=fields,
        timeout=10,
    )
    resp.raise_for_status()
