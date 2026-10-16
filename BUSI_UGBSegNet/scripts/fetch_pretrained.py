"""Download the required public ImageNet backbone only when absent; verify full SHA-256."""
import json
import urllib.request

from common import ROOT, file_hash


def main():
    folder = ROOT / "pretrained"
    metadata = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    target = folder / "efficientnet_b0_rwightman-7f5810bc.pth"
    if target.is_file():
        assert file_hash(target) == metadata["sha256"], "Existing pretrained weights have a different SHA-256."
        print("预训练权重已存在，SHA-256 正确，直接复用。")
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with urllib.request.urlopen(metadata["source"], timeout=120) as source, temporary.open("wb") as destination:
            while block := source.read(1024 * 1024):
                destination.write(block)
        assert file_hash(temporary) == metadata["sha256"], "Downloaded weights failed SHA-256 verification."
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print("预训练权重已下载并通过 SHA-256 检查。")


if __name__ == "__main__":
    main()
