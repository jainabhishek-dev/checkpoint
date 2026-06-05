"""Google Drive API operations — file reading, commenting, and image uploading.

Merges reader.py + commenter.py into a single module and consolidates the
duplicate _build_drive_service() helpers.
"""

import json
import re
from io import BytesIO

import os

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# ── MIME type maps ─────────────────────────────────────────────────────────────

SUPPORTED_TYPES = {
    "application/vnd.google-apps.document": "google_doc",
    "application/vnd.google-apps.presentation": "google_slides",
    "application/pdf": "pdf",
}

UNSUPPORTED_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        "Word (.docx) files are not supported directly. "
        "Please open the file in Google Drive, then go to File → Save as Google Docs, and share that link instead."
    ),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": (
        "PowerPoint (.pptx) files are not supported directly. "
        "Please open the file in Google Drive, then go to File → Save as Google Slides, and share that link instead."
    ),
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _build_credentials(token: dict) -> Credentials:
    return Credentials(
        token=token.get("access_token"),
        refresh_token=token.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    )


def _build_drive_service(creds: Credentials):
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _get_pdf_bytes_by_id(file_id: str, file_type: str, drive_service) -> bytes:
    """Export or download a file from Drive as raw PDF bytes."""
    if file_type in ("google_doc", "google_slides"):
        return (
            drive_service.files()
            .export(fileId=file_id, mimeType="application/pdf")
            .execute()
        )
    else:  # file_type == "pdf"
        return (
            drive_service.files()
            .get_media(fileId=file_id)
            .execute()
        )


def _format_comment(finding: dict, checkpoint_map: dict) -> str:
    """Format a finding as a Drive comment string."""
    cp = checkpoint_map.get(finding["checkpoint_id"], {})
    cp_name = cp.get("name", finding["checkpoint_id"])
    lines = [
        f"[{finding['checkpoint_id']}] {cp_name}",
        "",
        f"Location: {finding['location']}",
        f"Quoted text: \"{finding['quote']}\"",
        "",
        f"Issue: {finding['issue']}",
        f"Suggestion: {finding['suggestion']}",
    ]
    return "\n".join(lines)


def _add_comment_to_doc(drive_service, file_id: str, content: str) -> bool:
    """Add an unanchored comment to a Google Doc. Returns True on success."""
    try:
        drive_service.comments().create(
            fileId=file_id,
            body={"content": content},
            fields="id",
        ).execute()
        return True
    except Exception:
        return False


def _add_comment_to_slides(drive_service, file_id: str, content: str) -> bool:
    """Add a file-level comment to a Google Slides presentation. Returns True on success."""
    try:
        drive_service.comments().create(
            fileId=file_id,
            body={"content": content},
            fields="id",
        ).execute()
        return True
    except Exception:
        return False


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_file_id(url: str) -> str:
    """Extract Google Drive file ID from a share link."""
    patterns = [
        r"/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(
        "Could not extract a file ID from the URL you provided. "
        "Please paste the full Google Drive or Docs/Slides share link."
    )


def get_file_as_pdf(token: dict, drive_url: str) -> dict:
    """
    Export a Google Drive file as PDF bytes.

    Supports Google Docs, Google Slides, and PDF files.

    Returns:
        {"file_id", "title", "file_type", "pdf_bytes"}
        file_type is "google_doc" | "google_slides" | "pdf"
    """
    file_id = extract_file_id(drive_url)
    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)

    metadata = (
        drive_service.files()
        .get(fileId=file_id, fields="id,name,mimeType", supportsAllDrives=True)
        .execute()
    )
    mime_type = metadata.get("mimeType", "")
    title = metadata.get("name", "Untitled")

    if mime_type in UNSUPPORTED_TYPES:
        raise ValueError(UNSUPPORTED_TYPES[mime_type])

    if mime_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"This file type ({mime_type}) is not supported. "
            "Please provide a link to a Google Doc, Google Slides presentation, or PDF."
        )

    file_type = SUPPORTED_TYPES[mime_type]
    pdf_bytes = _get_pdf_bytes_by_id(file_id, file_type, drive_service)

    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError(
            f"No content was retrieved from '{title}'. "
            "The file may be empty or you may not have access to it."
        )

    return {
        "file_id": file_id,
        "title": title,
        "file_type": file_type,
        "pdf_bytes": pdf_bytes,
    }


def get_pdf_bytes_by_id(token: dict, file_id: str) -> dict:
    """
    Get PDF bytes for a file using its Drive ID (no URL parsing needed).

    Returns:
        {"file_type", "pdf_bytes"}
    """
    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)

    metadata = (
        drive_service.files()
        .get(fileId=file_id, fields="mimeType", supportsAllDrives=True)
        .execute()
    )
    mime_type = metadata.get("mimeType", "")

    if mime_type not in SUPPORTED_TYPES:
        raise ValueError(f"This file type ({mime_type}) is not supported.")

    file_type = SUPPORTED_TYPES[mime_type]
    pdf_bytes = _get_pdf_bytes_by_id(file_id, file_type, drive_service)

    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError("No content was retrieved. The file may be empty or inaccessible.")

    return {
        "file_type": file_type,
        "pdf_bytes": pdf_bytes,
    }


def create_drive_subfolder(token: dict, parent_folder_id: str, folder_name: str) -> str:
    """Create a subfolder inside parent_folder_id. Returns the new folder's Drive file ID."""
    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = drive_service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def upload_jpeg_to_drive(token: dict, folder_id: str, filename: str, image_bytes: bytes) -> str:
    """Upload a JPEG image to a Drive folder. Returns the uploaded file's Drive file ID.

    Attempts to set 'anyone can view' permission but silently continues if the
    domain policy blocks it — the authenticated proxy serves files to LEAD School
    users via OAuth token regardless of public sharing status.
    """
    from googleapiclient.http import MediaIoBaseUpload

    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)
    metadata = {"name": filename, "parents": [folder_id]}
    media = MediaIoBaseUpload(BytesIO(image_bytes), mimetype="image/jpeg")
    file = drive_service.files().create(
        body=metadata, media_body=media, fields="id"
    ).execute()
    file_id = file["id"]
    # Attempt to make the file publicly readable. This may fail if the domain
    # policy blocks external sharing — that's fine, the authenticated proxy works.
    try:
        drive_service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()
    except Exception as e:
        print(f"[drive] Could not set public permission on {file_id}: {e}")
    return file_id  # always returned regardless of permissions outcome


def fetch_drive_comments_with_pages(token: dict, file_id: str) -> list[dict]:
    """Fetch all unresolved comments from a Drive PDF and map each to its page number.

    Drive API comment anchor format for PDFs:
      anchor = '{"a": [{"kind": "drive#commentRegion", "page": N}]}'

    Returns:
        list of {"id", "content", "author", "page_num"}
        page_num is None for comments with no page anchor.
    """
    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)
    results = []
    page_token = None
    while True:
        resp = drive_service.comments().list(
            fileId=file_id,
            fields="comments(id,content,author(displayName),anchor,resolved),nextPageToken",
            pageSize=100,
            pageToken=page_token,
        ).execute()
        for c in resp.get("comments", []):
            if c.get("resolved"):
                continue
            page_num = None
            try:
                anchor = json.loads(c.get("anchor") or "{}")
                regions = anchor.get("a", [])
                if regions and "page" in regions[0]:
                    page_num = regions[0]["page"]  # 1-based
            except Exception:
                pass
            results.append({
                "id": c["id"],
                "content": c.get("content", ""),
                "author": c.get("author", {}).get("displayName", ""),
                "page_num": page_num,
            })
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


def extract_pdf_annotations(pdf_bytes: bytes) -> list[dict]:
    """Extract text annotations from a PDF using PyMuPDF.

    Returns list of {"content": str, "page_num": int} in page order (1-based).
    Only annotations with non-empty content are included. Used as a fallback to
    assign page numbers to Drive comments that lack an anchor (e.g. comments
    added in Adobe Acrobat and uploaded to Drive as a PDF).
    """
    import fitz

    results = []
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    try:
        for page_idx in range(len(doc)):
            for annot in doc[page_idx].annots():
                content = (annot.info.get("content") or "").strip()
                if content:
                    results.append({"content": content, "page_num": page_idx + 1})
    finally:
        doc.close()
    return results


def post_selected_comments(token: dict, file_data: dict, findings: list[dict], checkpoint_map: dict) -> int:
    """Post selected findings as Drive comments on the file.

    For PDFs, no comments are posted (Drive API does not support anchored comments on PDFs).
    Returns the number of comments successfully posted.
    """
    if file_data["file_type"] == "pdf":
        return 0

    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)
    file_id = file_data["file_id"]
    file_type = file_data["file_type"]
    posted = 0

    for finding in findings:
        comment_text = _format_comment(finding, checkpoint_map)
        if file_type == "google_doc":
            success = _add_comment_to_doc(drive_service, file_id, comment_text)
        elif file_type == "google_slides":
            success = _add_comment_to_slides(drive_service, file_id, comment_text)
        else:
            success = False
        if success:
            posted += 1

    return posted


def download_drive_image(token: dict, file_id: str) -> bytes:
    """Download a Drive file (JPEG page image) as raw bytes using the user's token."""
    creds = _build_credentials(token)
    drive_service = _build_drive_service(creds)
    request = drive_service.files().get_media(fileId=file_id)
    return request.execute()
