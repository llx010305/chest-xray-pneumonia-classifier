"""Download the two public Mendeley chest X-ray archives."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from src.common import ensure_dirs, load_config

CHUNK = 1024 * 1024
USER_AGENT = "cxr-pneumonia-classifier/1.0 (research reproduction; educational)"


def _destination(raw_dir: Path, source: dict) -> Path:
    folder = raw_dir / source["id"]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / source["file_name"]


def _looks_like_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist())
    except zipfile.BadZipFile:
        return False


def download_file(url: str, destination: Path) -> None:
    """Stream a URL to disk. A finished, valid zip is left untouched."""
    if _looks_like_zip(destination):
        print(f"already downloaded: {destination.name} ({destination.stat().st_size} bytes)")
        return

    partial = destination.with_suffix(destination.suffix + ".partial")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        if status == 200 and downloaded:
            downloaded = 0
            partial.write_bytes(b"")
        total = response.headers.get("Content-Length")
        total_bytes = int(total) + downloaded if total and status == 206 else (int(total) if total else None)
        mode = "ab" if downloaded and status == 206 else "wb"
        started = time.time()
        done = downloaded
        next_report = done
        with partial.open(mode) as handle:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                if done - next_report >= 20 * CHUNK or (total_bytes and done >= total_bytes):
                    elapsed = max(time.time() - started, 1e-6)
                    speed = (done - downloaded) / elapsed / (1024 * 1024)
                    if total_bytes:
                        print(f"  {destination.name}: {done / total_bytes:.1%} ({speed:.1f} MB/s)", flush=True)
                    else:
                        print(f"  {destination.name}: {done / (1024 * 1024):.0f} MB ({speed:.1f} MB/s)", flush=True)
                    next_report = done
    partial.replace(destination)
    if not _looks_like_zip(destination):
        raise RuntimeError(f"download is not a valid zip: {destination}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = load_config()
    paths = ensure_dirs()
    checksums = {}
    for source in config["sources"]:
        destination = _destination(paths["raw"], source)
        print(f"downloading {source['id']} from {source['url']}")
        download_file(source["download_url"], destination)
        checksums[source["id"]] = {
            "file": str(destination.relative_to(paths["raw"].parent.parent)),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "doi": source["doi"],
            "license": source["license"],
        }
        print(f"  sha256 {checksums[source['id']]['sha256']}")
    manifest = paths["raw"] / "checksums.json"
    manifest.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        raise
