import io
import logging
import re
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

GDRIVE_PATTERN = re.compile(
    r"https://drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:export=\w+&)?id=)([A-Za-z0-9_-]+)"
)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """Sliding-window character chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(data)
    elif ext == ".docx":
        return _extract_docx(data)
    else:
        return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def _extract_docx(data: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


async def fetch_url(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)


def extract_gdrive_file_id(url: str) -> Optional[str]:
    m = GDRIVE_PATTERN.search(url)
    return m.group(1) if m else None


async def fetch_gdrive(file_id: str) -> tuple[bytes, str]:
    """Download a Google Drive file. Returns (bytes, filename)."""
    export_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        resp = await client.get(export_url)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        # Large files trigger a virus-scan confirmation page
        if "text/html" in content_type:
            soup = BeautifulSoup(resp.text, "html.parser")
            token_input = soup.find("input", {"name": "confirm"})
            if token_input:
                token = token_input.get("value", "t")
            else:
                token = "t"
            confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
            resp = await client.get(confirm_url)
            resp.raise_for_status()

        # Extract filename from Content-Disposition
        cd = resp.headers.get("content-disposition", "")
        fname_match = re.search(r'filename[*]?=["\']?([^"\';\r\n]+)["\']?', cd)
        filename = fname_match.group(1).strip() if fname_match else f"{file_id}.pdf"
        return resp.content, filename
