"""Frozen data protocol, paths and atomic, inspectable outputs."""
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = "26e5f1d6d51b5ac36b338d1968df74f9024532c54f250417e8a609c2704a4387"
UPSTREAM_COMMIT = "20c8e4ea20691a49dd07c65cd1e3f44ef3eff2a4"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(array):
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    assert rows
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def manifest_rows(split=None):
    path = ROOT / "data" / "manifest.csv"
    assert file_hash(path) == MANIFEST_SHA256, "Frozen manifest changed."
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows if split is None else [row for row in rows if row["split"] == split]


def find_data_root(explicit=None):
    rows = manifest_rows()
    candidates = ([Path(explicit)] if explicit else [])
    if os.environ.get("BUSI_DATA_ROOT"):
        candidates.append(Path(os.environ["BUSI_DATA_ROOT"]))
    candidates += [ROOT / "data" / "prepared", ROOT.parent / "BUSI_nnUNet",
                   ROOT.parent.parent / "BUSI_nnUNet",
                   ROOT.parent.parent / "BUSI_UGBSegNet" / "data" / "prepared"]
    for candidate in candidates:
        if (candidate / rows[0]["image"]).is_file():
            return candidate.resolve()
    raise FileNotFoundError("未找到数据。AutoDL 请解压完整交付包；自定义位置设置 DATA_ROOT 或 BUSI_DATA_ROOT。")


def verify_data(data_root, deep=True):
    root = Path(data_root).resolve()
    rows = manifest_rows()
    assert len({row["id"] for row in rows}) == len(rows) == 1317
    counts = Counter(row["split"] for row in rows)
    assert dict(counts) == {"train": 453, "val": 65, "test": 129, "external": 670}
    split = json.loads((ROOT / "data" / "splits_final.json").read_text())[0]
    for name in ("train", "val"):
        assert set(split[name]) == {r["id"] for r in rows if r["split"] == name}
    grouped = defaultdict(set)
    for row in rows:
        grouped[row["image_sha256"]].add((row["dataset"], row["split"]))
        for key in ("image", "mask"):
            path = (root / row[key]).resolve()
            path.relative_to(root)
            assert path.is_file(), path
        if deep:
            with Image.open(root / row["image"]) as image:
                assert image.mode == "L"
                pixels = np.array(image)
            with Image.open(root / row["mask"]) as image:
                mask = np.array(image)
            assert pixels.shape == mask.shape == (int(row["height"]), int(row["width"]))
            assert set(np.unique(mask)) <= {0, 1}
            assert int(mask.sum()) == int(row["foreground_pixels"])
            assert pixel_hash(pixels) == row["image_sha256"], row["id"]
            assert pixel_hash(mask) == row["mask_sha256"], row["id"]
    assert all(len(splits) == 1 for splits in grouped.values()), "Exact duplicate crosses a split."
    return {"status": "PASS", "pairs": len(rows), "deep_pixel_checks": deep,
            "counts": dict(counts), "manifest_sha256": MANIFEST_SHA256, "data_root": str(root)}


def overlap_metrics(prediction, truth):
    pred, target = np.asarray(prediction, bool), np.asarray(truth, bool)
    assert pred.shape == target.shape
    tp, fp = int((pred & target).sum()), int((pred & ~target).sum())
    fn, tn = int((~pred & target).sum()), int((~pred & ~target).sum())
    return {"dice": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            "iou": tp/(tp+fp+fn) if tp+fp+fn else 1.,
            "sensitivity": tp/(tp+fn) if tp+fn else None,
            "specificity": tn/(tn+fp) if tn+fp else None,
            "pixel_accuracy": (tp+tn)/pred.size,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "predicted_foreground_pixels": int(pred.sum())}
