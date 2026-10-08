"""Pure sampling and split helpers. No image I/O, so they can be unit tested."""

from __future__ import annotations

import random
from collections import defaultdict

from sklearn.model_selection import train_test_split

LABELS = ("NORMAL", "PNEUMONIA")


def class_counts(records: list[dict], key: str = "label") -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for record in records:
        counts[record[key]] = counts.get(record[key], 0) + 1
    return counts


def choose_class_targets(available: dict[str, int], total: int, max_ratio: float) -> dict[str, int]:
    """Pick per-class counts that sum to `total` and stay within `max_ratio`.

    Among feasible pairs, the one closest to a 50/50 split wins.
    """
    best: tuple[int, int, int] | None = None
    for pneumonia in range(total + 1):
        normal = total - pneumonia
        if normal == 0 or pneumonia == 0:
            continue
        if normal > available.get("NORMAL", 0) or pneumonia > available.get("PNEUMONIA", 0):
            continue
        small = min(normal, pneumonia)
        large = max(normal, pneumonia)
        if large / small > max_ratio + 1e-9:
            continue
        imbalance = abs(normal - pneumonia)
        candidate = (imbalance, pneumonia, normal)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError(
            f"cannot build {total} images within ratio {max_ratio} from available counts {available}"
        )
    return {"PNEUMONIA": best[1], "NORMAL": best[2]}


def _round_robin(pools: dict[str, list[dict]], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    ordered = sorted(pools)
    mutable = {source: records[:] for source, records in pools.items()}
    for records in mutable.values():
        rng.shuffle(records)
    chosen: list[dict] = []
    while len(chosen) < count:
        progressed = False
        for source in ordered:
            if mutable[source] and len(chosen) < count:
                chosen.append(mutable[source].pop())
                progressed = True
        if not progressed:
            raise RuntimeError("class pool ran out during round-robin sampling")
    return chosen


def sample_working_set(
    records: list[dict],
    total: int = 3000,
    max_ratio: float = 1.5,
    min_per_source: int = 400,
    seed: int = 42,
) -> list[dict]:
    """Sample a fixed-size, class-balanced working set that uses every source."""
    sources = sorted({record["source"] for record in records})
    if len(sources) < 2:
        raise RuntimeError("at least two sources are required")

    pools: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        pools[(record["label"], record["source"])].append(record)

    available = {
        label: sum(len(pools[(label, source)]) for source in sources) for label in LABELS
    }
    targets = choose_class_targets(available, total, max_ratio)
    chosen: list[dict] = []
    for offset, label in enumerate(LABELS):
        label_pools = {source: pools[(label, source)] for source in sources}
        chosen.extend(_round_robin(label_pools, targets[label], seed + offset))

    per_source: dict[str, int] = defaultdict(int)
    for record in chosen:
        per_source[record["source"]] += 1
    short = {source: count for source, count in per_source.items() if count < min_per_source}
    if short or len(per_source) < len(sources):
        raise RuntimeError(
            f"sampling left a source below the minimum of {min_per_source}: {dict(per_source)}"
        )

    counts = class_counts(chosen)
    small = min(counts.values())
    large = max(counts.values())
    if len(chosen) != total or large / small > max_ratio + 1e-9:
        raise RuntimeError(f"sample failed the balance contract: {counts}")
    return chosen


def drop_duplicates(records: list[dict], keys: tuple[str, ...] = ("sha256", "tiny_sig")) -> tuple[list[dict], list[dict]]:
    """Drop later records that repeat an earlier hash. Caller controls priority order."""
    seen = {key: {} for key in keys}
    kept: list[dict] = []
    dropped: list[dict] = []
    for record in records:
        reason = None
        for key in keys:
            value = record.get(key)
            if value and value in seen[key]:
                reason = key
                break
        if reason:
            dropped.append({"image_id": record.get("image_id"), "reason": reason, "source": record.get("source")})
            continue
        for key in keys:
            value = record.get(key)
            if value:
                seen[key][value] = record.get("image_id")
        kept.append(record)
    return kept, dropped


def assign_splits(records: list[dict], seed: int = 42) -> list[dict]:
    """Assign train/val/test by patient group so one patient stays in one split.

    Group rows are stratified by label. Image counts therefore land near
    70/15/15, not exactly on those fractions when group sizes differ.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["patient_id"]].append(record)

    group_rows = []
    for patient_id, members in groups.items():
        labels = {member["label"] for member in members}
        label = members[0]["label"] if len(labels) == 1 else max(
            labels, key=lambda item: sum(member["label"] == item for member in members)
        )
        group_rows.append({"patient_id": patient_id, "label": label, "members": members})

    labels = [row["label"] for row in group_rows]
    trainval, test = train_test_split(
        group_rows, test_size=0.15, random_state=seed, stratify=labels
    )
    trainval_labels = [row["label"] for row in trainval]
    train, val = train_test_split(
        trainval,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=trainval_labels,
    )
    split_of = {}
    for name, rows in (("train", train), ("val", val), ("test", test)):
        for row in rows:
            split_of[row["patient_id"]] = name

    assigned = []
    for record in records:
        copied = dict(record)
        copied["split"] = split_of[record["patient_id"]]
        assigned.append(copied)
    return assigned
