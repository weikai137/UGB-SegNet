"""Pinned upstream network/loss, with explicit offline weights and 0/1 mask support."""
import ast
import importlib.util
import os
import random

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from common import ROOT, file_hash


def build_model(pretrained=True):
    path = ROOT / "vendor" / "UGB-SegNet" / "ugbnet_segmentation_improved.py"
    assert file_hash(path) == "41b4f67bb36383e6e78f574661a84144b2c4cf5c2b1639831521740202389c69"
    spec = importlib.util.spec_from_file_location("ugb_pinned_upstream", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_factory = module.efficientnet_b0
    def offline_factory(**kwargs):
        backbone = original_factory(weights=None)
        if pretrained:
            asset = ROOT / "pretrained" / "efficientnet_b0_rwightman-7f5810bc.pth"
            metadata = __import__("json").loads((ROOT / "pretrained" / "provenance.json").read_text())
            assert file_hash(asset) == metadata["sha256"], "Pretrained weights changed."
            backbone.load_state_dict(torch.load(asset, map_location="cpu", weights_only=True))
        return backbone
    module.efficientnet_b0 = offline_factory
    return module.UGBSegNetImproved()  # Architecture, dropout and checkpointing unchanged.


def build_loss():
    # Extract the exact class without importing the author's plotting/training driver.
    path = ROOT / "vendor" / "UGB-SegNet" / "run_5fold_cv.py"
    metadata = __import__("json").loads((ROOT / "vendor" / "provenance.json").read_text())
    assert file_hash(path) == metadata["files"][path.name]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SegLoss")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["SegLoss"]()


class BusiDataset(Dataset):
    def __init__(self, rows, data_root, augment=False, size=224, seed=42):
        from pathlib import Path
        self.rows, self.data_root, self.size = rows, Path(data_root), size
        self.aug = None
        self.seed = seed
        self.to_tensor = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        if augment:
            import albumentations as A
            import cv2
            cv2.setNumThreads(0)
            self.aug = A.Compose([
                A.HorizontalFlip(p=.5), A.VerticalFlip(p=.3), A.Rotate(limit=30, p=.5),
                A.RandomBrightnessContrast(p=.4), A.GaussNoise(p=.3),
                A.ElasticTransform(p=.2),
                A.RandomResizedCrop(size=(size, size), scale=(.8, 1.), p=.3)])
            self.aug.set_random_seed(seed)

    def set_epoch(self, epoch):
        if self.aug is not None:
            self.aug.set_random_seed(self.seed + epoch * 1009)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.data_root / row["image"]) as image:
            pixels = np.array(image.convert("RGB").resize((self.size, self.size), Image.Resampling.BICUBIC))
        with Image.open(self.data_root / row["mask"]) as image:
            mask = (np.array(image.resize((self.size, self.size), Image.Resampling.NEAREST)) > 0).astype(np.uint8)
        if self.aug is not None:
            transformed = self.aug(image=pixels, mask=mask)
            pixels, mask = transformed["image"], transformed["mask"]
        return self.to_tensor(Image.fromarray(pixels)), torch.from_numpy(mask.copy()).float()[None]


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    dataset = torch.utils.data.get_worker_info().dataset
    if dataset.aug is not None:
        dataset.aug.set_random_seed(seed)
