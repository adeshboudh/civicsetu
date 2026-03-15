from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import structlog

from civicsetu.config.settings import get_settings

log = structlog.get_logger(__name__)
settings = get_settings()


class Downloader:
    """
    Downloads PDFs from government URLs to local data/raw/ cache.
    Skips re-download if file already exists and hash matches.
    All methods are synchronous — downloads are one-shot at ingestion time.
    """

    TIMEOUT = 60
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; CivicSetu/1.0; "
            "+https://github.com/civicsetu)"
        )
    }

    @staticmethod
    def download(
        url: str,
        dest_dir: str | Path,
        filename: str | None = None,
        force: bool = False,
    ) -> Path:
        """
        Download a PDF to dest_dir.
        Returns path to the local file.
        Skips download if file exists unless force=True.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = url.split("/")[-1].split("?")[0]
            if not filename.endswith(".pdf"):
                filename += ".pdf"

        dest_path = dest_dir / filename

        if dest_path.exists() and not force:
            log.info("download_skipped_cached", path=str(dest_path))
            return dest_path

        log.info("downloading", url=url, dest=str(dest_path))

        try:
            with httpx.Client(
                timeout=Downloader.TIMEOUT,
                headers=Downloader.HEADERS,
                follow_redirects=True,
            ) as client:
                response = client.get(url)
                response.raise_for_status()

        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HTTP {e.response.status_code} downloading {url}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(f"Timeout downloading {url}") from e

        content = response.content
        if not content:
            raise RuntimeError(f"Empty response from {url}")

        dest_path.write_bytes(content)

        md5 = hashlib.md5(content).hexdigest()
        log.info(
            "download_complete",
            path=str(dest_path),
            size_kb=len(content) // 1024,
            md5=md5,
        )
        return dest_path

    @staticmethod
    def download_many(
        sources: list[dict],
        base_dir: str | Path = "data/raw",
        force: bool = False,
    ) -> dict[str, Path]:
        """
        Batch download from a list of source dicts.
        Each dict: {"url": ..., "subdir": ..., "filename": ...}
        Returns {filename: local_path}
        """
        results = {}
        for source in sources:
            dest_dir = Path(base_dir) / source.get("subdir", "")
            path = Downloader.download(
                url=source["url"],
                dest_dir=dest_dir,
                filename=source.get("filename"),
                force=force,
            )
            results[source.get("filename", path.name)] = path
            log.info("batch_download_progress", done=len(results), total=len(sources))

        return results
