"""Build a self-contained upload ZIP from the shared frozen prepared data."""
import argparse
import json
import zipfile
from pathlib import Path

from common import ROOT, MANIFEST_SHA256, file_hash, find_data_root, manifest_rows


def package(output=None, data_root=None):
    output = Path(output or ROOT.parent / "BUSI_UGBSegNet_AutoDL.zip").resolve()
    if output.exists():
        raise FileExistsError(f"已有交付包，不覆盖：{output}")
    data_root = find_data_root(data_root)
    files = []
    for folder in ("scripts", "vendor", "pretrained", "data", "reports"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and not any(part in {"__pycache__", "prepared", ".ipynb_checkpoints", "_qa_work"} for part in path.relative_to(ROOT).parts) and not path.name.endswith((".tmp", ".pyc")):
                files.append((path, ROOT.name+"/"+path.relative_to(ROOT).as_posix()))
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix in {".ipynb", ".md", ".txt", ".yml", ".sh"}:
            files.append((path, ROOT.name+"/"+path.name))
    seen_data = set()
    for row in manifest_rows():
        for key in ("image", "mask"):
            relative = row[key]
            if relative not in seen_data:
                path = (data_root / relative).resolve()
                path.relative_to(data_root)
                assert path.is_file(), path
                files.append((path, ROOT.name+"/data/prepared/"+relative))
                seen_data.add(relative)
    assert len(seen_data) == 2634
    names = [name for _, name in files]
    assert len(names) == len(set(names))
    temporary = output.with_suffix(output.suffix+".tmp")
    try:
        # PNGs and pretrained tensors are already compact; STORE avoids CPU-heavy recompression.
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path, name in files:
                archive.write(path, name)
        with zipfile.ZipFile(temporary) as archive:
            assert archive.testzip() is None
            assert len(archive.namelist()) == len(files)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    result = {"zip": str(output), "bytes": output.stat().st_size, "sha256": file_hash(output),
              "file_count": len(files), "prepared_pairs": 1317, "manifest_sha256": MANIFEST_SHA256,
              "includes": ["notebook", "source", "pinned upstream", "ImageNet backbone weights", "frozen train/val/test/external data"],
              "trained_model_included": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    package(args.output, args.data_root)
