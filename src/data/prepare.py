"""Extract both archives, drop duplicate films, and sample the 3000-image working set."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.common import ROOT, ensure_dirs, load_config
from src.data.sample import (
    LABELS,
    assign_splits,
    class_counts,
    duplicate_clusters,
    resolve_duplicates,
    sample_working_set,
    zscore_thumbnail,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_FROM_DIR = {
    "normal": "NORMAL",
    "pneumonia": "PNEUMONIA",
}


def _safe_extract(zip_path: Path, dest: Path, nested: bool = True) -> None:
    """Extract images into short split/label/filename paths.

    The published archives repeat long dataset titles, which can exceed the
    legacy Windows path limit. Keeping the last three path parts preserves
    the class folder without those titles.
    """
    dest.mkdir(parents=True, exist_ok=True)
    nested_zips: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            parts = Path(info.filename).parts
            if info.is_dir() or not parts or ".." in parts or info.filename.startswith("/"):
                if ".." in parts or info.filename.startswith("/"):
                    raise RuntimeError(f"unsafe zip path: {info.filename}")
                continue
            filename = parts[-1]
            if "__MACOSX" in parts or filename.startswith("._") or filename == ".DS_Store":
                continue
            suffix = Path(filename).suffix.lower()
            if suffix == ".zip" and nested:
                target = dest / f"nested_{len(nested_zips)}.zip"
                _extract_member(archive, info, target)
                nested_zips.append(target)
                continue
            if suffix not in IMAGE_SUFFIXES:
                continue
            label = parts[-2] if len(parts) >= 2 else "unknown"
            split_name = parts[-3] if len(parts) >= 3 else "all"
            target = dest / split_name / label / filename
            if target.exists():
                target = target.with_stem(f"{target.stem}_{len(nested_zips)}")
            _extract_member(archive, info, target)
    for index, nested_zip in enumerate(nested_zips):
        _safe_extract(nested_zip, dest / f"nested_{index}", nested=False)
        nested_zip.unlink(missing_ok=True)


def _extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, target.open("wb") as handle:
        shutil.copyfileobj(source, handle)


def _label_for(path: Path) -> str | None:
    for parent in path.parents:
        mapped = LABEL_FROM_DIR.get(parent.name.lower())
        if mapped:
            return mapped
    return None


def _patient_id(source: str, path: Path) -> str:
    stem = path.stem
    if source == "kermany":
        person = re.search(r"person(\d+)", stem, re.IGNORECASE)
        if person:
            return f"kermany-person-{person.group(1)}"
        image = re.search(r"IM-(\d+)", stem, re.IGNORECASE)
        if image:
            return f"kermany-im-{image.group(1)}"
    return f"{source}-{stem}"


def _inventory_source(source_id: str, root: Path, thumb_size: int) -> list[dict]:
    records = []
    skipped = 0
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and "__MACOSX" not in path.parts
        and not path.name.startswith("._")
    ]
    for index, path in enumerate(sorted(paths)):
        label = _label_for(path)
        if label not in LABELS:
            skipped += 1
            continue
        payload = path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if decoded is None or decoded.size == 0:
            skipped += 1
            continue
        tiny = cv2.resize(decoded, (16, 16), interpolation=cv2.INTER_AREA)
        thumb = cv2.resize(decoded, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        records.append(
            {
                "source": source_id,
                "label": label,
                "patient_id": _patient_id(source_id, path),
                "original_path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "tiny_sig": tiny.tobytes().hex(),
                "thumb": zscore_thumbnail(thumb),
                "suffix": path.suffix.lower(),
            }
        )
        if index and index % 500 == 0:
            print(f"  inventoried {index} files from {source_id}", flush=True)
    print(f"  {source_id}: {len(records)} labeled images, {skipped} skipped")
    return records


def _write_manifest(path: Path, records: list[dict]) -> None:
    fieldnames = [
        "image_id",
        "source",
        "label",
        "patient_id",
        "split",
        "sha256",
        "archive_member",
        "selected_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _write_duplicate_report(path: Path, records: list[dict], clusters: list[list[int]], extract_root: Path) -> None:
    """One row per image that belongs to a duplicate cluster, with the decision taken."""
    fieldnames = ["cluster", "decision", "source", "label", "sha256", "archive_member"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cluster_id, cluster in enumerate(clusters):
            conflict = len({records[index]["label"] for index in cluster}) > 1
            for position, index in enumerate(cluster):
                record = records[index]
                if conflict:
                    decision = "dropped_label_conflict"
                else:
                    decision = "kept" if position == 0 else "dropped_duplicate"
                writer.writerow(
                    {
                        "cluster": cluster_id,
                        "decision": decision,
                        "source": record["source"],
                        "label": record["label"],
                        "sha256": record["sha256"],
                        "archive_member": Path(record["original_path"]).relative_to(extract_root).as_posix(),
                    }
                )


def _cluster_types(records: list[dict], clusters: list[list[int]]) -> dict[str, int]:
    """Count clusters by the source/label of their members, e.g. 'chittagong:PNEUMONIA + kermany:NORMAL'."""
    counter = Counter()
    for cluster in clusters:
        kinds = sorted({f"{records[index]['source']}:{records[index]['label']}" for index in cluster})
        counter[" + ".join(kinds)] += 1
    return dict(sorted(counter.items()))


def _count_by(records: list[dict], *keys: str) -> dict:
    counter = Counter()
    for record in records:
        counter["|".join(str(record[key]) for key in keys)] += 1
    return dict(sorted(counter.items()))


def prepare(config: dict | None = None) -> dict:
    config = config or load_config()
    paths = ensure_dirs()
    extract_root = paths["interim"] / "extracted"
    selected_root = paths["interim"] / "selected"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    if selected_root.exists():
        shutil.rmtree(selected_root)
    selected_root.mkdir(parents=True)

    pooled: list[dict] = []
    for source in config["sources"]:
        zip_path = paths["raw"] / source["id"] / source["file_name"]
        if not zip_path.exists():
            raise FileNotFoundError(f"missing archive for {source['id']}: {zip_path}. Run download first.")
        dest = extract_root / source["id"]
        print(f"extracting {zip_path.name}")
        _safe_extract(zip_path, dest)
        pooled.extend(_inventory_source(source["id"], dest, int(config.get("dedup_thumb_size", 32))))

    before = _count_by(pooled, "source", "label")
    # Priority order: Kermany before Chittagong, then by extracted path. The
    # first copy in a duplicate cluster is the one kept.
    pooled.sort(key=lambda record: (record["source"] != "kermany", record["original_path"]))
    vectors = np.stack([record["thumb"] for record in pooled])
    clusters = duplicate_clusters(pooled, vectors, threshold=float(config.get("dedup_correlation", 0.995)))
    unique, dropped = resolve_duplicates(pooled, clusters)
    for record in pooled:
        record.pop("thumb", None)
    conflict_clusters = sum(1 for cluster in clusters if len({pooled[i]["label"] for i in cluster}) > 1)
    print(
        f"found {len(clusters)} duplicate clusters ({conflict_clusters} with conflicting labels); "
        f"dropped {len(dropped)} images; {len(unique)} remain"
    )
    _write_duplicate_report(paths["reports"] / "duplicate_clusters.csv", pooled, clusters, extract_root)

    # Only the working-set sources are sampled for train/val/test. Every unique
    # image from an external source is kept aside as an external check.
    working_sources = set(config.get("working_set_sources") or [s["id"] for s in config["sources"]])
    pool = [record for record in unique if record["source"] in working_sources]
    external = [record for record in unique if record["source"] not in working_sources]

    chosen = sample_working_set(
        pool,
        total=int(config["target_images"]),
        max_ratio=float(config["max_class_ratio"]),
        min_per_source=int(config["min_images_per_source"]),
        seed=int(config["seed"]),
    )
    chosen = assign_splits(chosen, seed=int(config["seed"]))
    for record in external:
        record["split"] = "external"
    chosen.extend(external)
    chosen.sort(key=lambda record: (record["split"], record["label"], record["source"], record["sha256"]))

    final = []
    for index, record in enumerate(chosen, start=1):
        image_id = f"{record['source']}_{record['label']}_{index:04d}"
        suffix = record["suffix"] if record["suffix"] in IMAGE_SUFFIXES else ".jpg"
        selected_path = selected_root / f"{image_id}{suffix}"
        shutil.copy2(record["original_path"], selected_path)
        copied = dict(record)
        copied["image_id"] = image_id
        copied["archive_member"] = Path(record["original_path"]).relative_to(extract_root).as_posix()
        copied["selected_path"] = selected_path.relative_to(ROOT).as_posix()
        final.append(copied)

    manifest_path = paths["reports"] / "manifest.csv"
    _write_manifest(manifest_path, final)
    summary = {
        "target_images": int(config["target_images"]),
        "max_class_ratio": float(config["max_class_ratio"]),
        "seed": int(config["seed"]),
        "before_dedup": before,
        "dedup_correlation": float(config.get("dedup_correlation", 0.995)),
        "dedup_thumb_size": int(config.get("dedup_thumb_size", 32)),
        "duplicate_clusters": len(clusters),
        "label_conflict_clusters": conflict_clusters,
        "duplicate_cluster_types": _cluster_types(pooled, clusters),
        "dropped_duplicates": len(dropped),
        "duplicate_reasons": dict(Counter(item["reason"] for item in dropped)),
        "after_dedup": _count_by(unique, "source", "label"),
        "working_set_sources": sorted(working_sources),
        "working_set": _count_by([r for r in final if r["split"] != "external"], "source", "label"),
        "external_set": _count_by([r for r in final if r["split"] == "external"], "source", "label"),
        "splits": _count_by(final, "split", "label"),
        "class_counts": class_counts([r for r in final if r["split"] != "external"]),
        "source_counts": dict(Counter(record["source"] for record in final)),
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
    }
    summary_path = paths["reports"] / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.rmtree(extract_root)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    prepare()


if __name__ == "__main__":
    main()
