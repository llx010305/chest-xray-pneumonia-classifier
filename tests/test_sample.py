"""Balance contract for sampling, de-duplication, and patient-grouped splits."""

from __future__ import annotations

import unittest
from collections import Counter

from src.data.sample import (
    assign_splits,
    choose_class_targets,
    class_counts,
    drop_duplicates,
    sample_working_set,
)


def _fake_records(per_pool: dict[tuple[str, str], int]) -> list[dict]:
    records = []
    for (source, label), count in per_pool.items():
        for index in range(count):
            records.append(
                {
                    "source": source,
                    "label": label,
                    "patient_id": f"{source}-{label}-{index}",
                    "sha256": f"{source}-{label}-{index}",
                    "tiny_sig": f"sig-{source}-{label}-{index}",
                }
            )
    return records


class ChooseClassTargetsTest(unittest.TestCase):
    def test_balanced_when_both_classes_have_enough(self):
        targets = choose_class_targets({"NORMAL": 5000, "PNEUMONIA": 5000}, total=3000, max_ratio=1.5)
        self.assertEqual(targets, {"NORMAL": 1500, "PNEUMONIA": 1500})

    def test_closest_to_balanced_when_one_class_is_short(self):
        targets = choose_class_targets({"NORMAL": 1300, "PNEUMONIA": 5000}, total=3000, max_ratio=1.5)
        self.assertEqual(targets["NORMAL"], 1300)
        self.assertEqual(targets["PNEUMONIA"], 1700)
        self.assertLessEqual(max(targets.values()) / min(targets.values()), 1.5)

    def test_raises_when_ratio_cannot_be_met(self):
        with self.assertRaises(RuntimeError):
            choose_class_targets({"NORMAL": 500, "PNEUMONIA": 5000}, total=3000, max_ratio=1.5)


class SampleWorkingSetTest(unittest.TestCase):
    def setUp(self):
        self.records = _fake_records(
            {
                ("kermany", "NORMAL"): 1500,
                ("kermany", "PNEUMONIA"): 4000,
                ("chittagong", "NORMAL"): 1200,
                ("chittagong", "PNEUMONIA"): 800,
            }
        )

    def test_exact_size_and_balance(self):
        chosen = sample_working_set(self.records, total=3000, max_ratio=1.5, min_per_source=400, seed=42)
        self.assertEqual(len(chosen), 3000)
        self.assertEqual(class_counts(chosen), {"NORMAL": 1500, "PNEUMONIA": 1500})

    def test_every_source_contributes(self):
        chosen = sample_working_set(self.records, total=3000, max_ratio=1.5, min_per_source=400, seed=42)
        per_source = Counter(record["source"] for record in chosen)
        self.assertEqual(set(per_source), {"kermany", "chittagong"})
        self.assertTrue(all(count >= 400 for count in per_source.values()))

    def test_no_record_is_chosen_twice(self):
        chosen = sample_working_set(self.records, total=3000, seed=42)
        self.assertEqual(len({record["sha256"] for record in chosen}), len(chosen))

    def test_same_seed_gives_same_sample(self):
        first = sample_working_set(self.records, total=3000, seed=42)
        second = sample_working_set(self.records, total=3000, seed=42)
        self.assertEqual([r["sha256"] for r in first], [r["sha256"] for r in second])

    def test_needs_two_sources(self):
        one_source = [record for record in self.records if record["source"] == "kermany"]
        with self.assertRaises(RuntimeError):
            sample_working_set(one_source, total=3000, seed=42)


class DropDuplicatesTest(unittest.TestCase):
    def test_first_copy_wins(self):
        records = [
            {"source": "kermany", "sha256": "a", "tiny_sig": "x"},
            {"source": "chittagong", "sha256": "a", "tiny_sig": "y"},
            {"source": "chittagong", "sha256": "b", "tiny_sig": "x"},
            {"source": "chittagong", "sha256": "c", "tiny_sig": "z"},
        ]
        kept, dropped = drop_duplicates(records)
        self.assertEqual([record["sha256"] for record in kept], ["a", "c"])
        self.assertEqual(kept[0]["source"], "kermany")
        self.assertEqual([item["reason"] for item in dropped], ["sha256", "tiny_sig"])


class AssignSplitsTest(unittest.TestCase):
    def setUp(self):
        records = _fake_records({("kermany", "NORMAL"): 300, ("kermany", "PNEUMONIA"): 300})
        # Give some patients several images, as Kermany person ids do.
        for index, record in enumerate(records):
            record["patient_id"] = f"{record['label']}-patient-{index // 3}"
        self.records = records

    def test_patient_never_crosses_splits(self):
        assigned = assign_splits(self.records, seed=42)
        split_of: dict[str, set[str]] = {}
        for record in assigned:
            split_of.setdefault(record["patient_id"], set()).add(record["split"])
        self.assertTrue(all(len(splits) == 1 for splits in split_of.values()))

    def test_split_fractions_near_70_15_15(self):
        assigned = assign_splits(self.records, seed=42)
        counts = Counter(record["split"] for record in assigned)
        total = len(assigned)
        self.assertAlmostEqual(counts["train"] / total, 0.70, delta=0.03)
        self.assertAlmostEqual(counts["val"] / total, 0.15, delta=0.03)
        self.assertAlmostEqual(counts["test"] / total, 0.15, delta=0.03)

    def test_every_split_has_both_classes(self):
        assigned = assign_splits(self.records, seed=42)
        for name in ("train", "val", "test"):
            labels = {record["label"] for record in assigned if record["split"] == name}
            self.assertEqual(labels, {"NORMAL", "PNEUMONIA"})


if __name__ == "__main__":
    unittest.main()
