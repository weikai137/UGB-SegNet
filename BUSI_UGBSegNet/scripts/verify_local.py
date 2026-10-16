"""CPU acceptance checks. No benchmark training and no internal/external test inference."""
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import ast
import gc
import json
import shutil
import sys
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import ROOT, MANIFEST_SHA256, UPSTREAM_COMMIT, find_data_root, manifest_rows, overlap_metrics, verify_data, write_json
from ugb_adapter import BusiDataset, build_loss, build_model
from training import TrainConfig, atomic_torch_save, train_model
from evaluation import aggregate, evaluate_checkpoint, hd95_pixels


class ResumeProbe(nn.Module):
    """Small model isolates checkpoint/optimizer/augmentation resume correctness."""
    def __init__(self, pretrained=False):
        super().__init__()
        self.body = nn.Conv2d(3, 4, 1)
        self.head = nn.Conv2d(4, 1, 1)
    def forward(self, image):
        logits = self.head(torch.relu(self.body(image)))
        return {"main": logits, "ds3": logits, "ds2": logits, "ds1": logits}
    def freeze_backbone(self, n=4):
        for parameter in self.body.parameters(): parameter.requires_grad = False
    def unfreeze_all(self):
        for parameter in self.parameters(): parameter.requires_grad = True


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data_root = find_data_root()
    checks = []
    work = ROOT / "_qa_work"
    assert not work.exists(), "Preserve previous unfinished QA work for inspection."
    work.mkdir()
    try:
        for source in (ROOT / "scripts").glob("*.py"):
            ast.parse(source.read_text(encoding="utf-8"))
        audit = verify_data(data_root, deep=True)
        checks.append("1317 frozen image/mask pairs reopened: binary labels, shapes and pixel hashes match")
        truth = np.array([[1, 1], [0, 0]], dtype=bool)
        prediction = np.array([[1, 0], [1, 0]], dtype=bool)
        assert overlap_metrics(prediction, truth)["dice"] == .5
        assert np.isclose(overlap_metrics(prediction, truth)["iou"], 1/3)
        assert hd95_pixels(truth, truth) == 0
        assert hd95_pixels(np.zeros_like(truth), truth) is None
        normal = {"class": "normal", **overlap_metrics(prediction, np.zeros_like(truth)), "hd95_pixels": None}
        grouped = aggregate([normal])
        assert grouped["lesion_macro"]["n"] == 0 and grouped["normal_false_positive_rate_any_pixel"] == 1
        checks.append("Known-answer overlap/HD95/normal-image handling")
        train_rows, val_rows = manifest_rows("train")[:2], manifest_rows("val")[:2]
        dataset = BusiDataset(train_rows, data_root, augment=True)
        dataset.set_epoch(3)
        image, mask = dataset[0]
        assert image.shape == (3, 224, 224) and mask.shape == (1, 224, 224)
        assert set(mask.unique().tolist()) <= {0., 1.} and bool(mask.any())
        # Real pretrained UGB model, real image/mask, exact upstream loss, real CPU update.
        print("Checking real UGB-SegNet CPU forward/backward...", flush=True)
        torch.manual_seed(42)
        model = build_model(pretrained=True).cpu().train()
        criterion = build_loss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        result = model(image[None])
        assert {"main", "ds3", "ds2", "ds1", "uncertainty"} <= set(result)
        assert result["main"].shape == (1, 1, 224, 224)
        loss = criterion(result, mask[None], is_ugbnet=True)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        assert any(bool(torch.count_nonzero(gradient)) for gradient in gradients)
        optimizer.step()
        real_check = {"parameters": sum(parameter.numel() for parameter in model.parameters()),
                      "input_shape": [1, 3, 224, 224], "output_shape": list(result["main"].shape),
                      "loss_is_finite": True, "gradients_finite_and_nonzero": True,
                      "device": "cpu", "pretrained_asset_used": True}
        real_run = work / "real_ugb"
        real_run.mkdir()
        atomic_torch_save({"model": model.state_dict(), "epoch": 1, "best_val_dice_224": 0.,
            "manifest_sha256": MANIFEST_SHA256, "upstream_commit": UPSTREAM_COMMIT,
            "smoke": True}, real_run / "checkpoint_best.pth")
        del model, optimizer, result, gradients, criterion, loss
        gc.collect()
        checks.append("Real pinned pretrained UGB forward/backward/SGD update with finite nonzero gradients")
        evaluated = evaluate_checkpoint(data_root, real_run, "val", device="cpu", allow_smoke=True,
            rows_override=val_rows, output_base=work / "evaluation")
        evaluation = json.loads((evaluated / "summary.json").read_text())
        assert evaluation["status"] == "SMOKE_ONLY" and evaluation["n"] == 2
        assert len(list((evaluated / "predictions_native").glob("*.png"))) == 2
        assert (evaluated / "report.html").is_file() and (evaluated / "native_per_image.csv").is_file()
        checks.append("Two validation-only smoke images: native and 224 prediction, metrics, uncertainty, panels and report")
        # Check epoch-1 -> phase transition -> epoch-2 resume with exact CPU replay.
        config = TrainConfig(epochs=3, phase1_end=1, batch_size=2, workers=0,
            cpu_threads=2, selection_every=1, patience=999)
        kwargs = dict(config=config, device="cpu", smoke=True, model_factory=ResumeProbe,
                      rows_override=(train_rows, val_rows))
        print("Checking uninterrupted vs resumed phase-1/phase-2 training...", flush=True)
        train_model(data_root, work / "uninterrupted", **kwargs)
        train_model(data_root, work / "resumed", stop_after_epoch=1, **kwargs)
        train_model(data_root, work / "resumed", resume=True, stop_after_epoch=2, **kwargs)
        train_model(data_root, work / "resumed", resume=True, **kwargs)
        whole = torch.load(work / "uninterrupted" / "checkpoint_latest.pth", map_location="cpu", weights_only=False)
        resumed = torch.load(work / "resumed" / "checkpoint_latest.pth", map_location="cpu", weights_only=False)
        assert whole["epoch"] == resumed["epoch"] == 3 and resumed["phase"] == 2
        assert all(torch.equal(value, resumed["model"][name]) for name, value in whole["model"].items())
        for before, after in zip(whole["history"], resumed["history"]):
            for key in ("phase", "train_loss", "val_loss", "val_dice_224", "lr_used", "lr_next"):
                assert before[key] == after[key], (key, before[key], after[key])
        completed = train_model(data_root, work / "resumed", resume=True, **kwargs)
        assert completed["completed_epochs"] == 3
        checks.append("Exact CPU replay after epoch 1 and epoch 2: same weights, losses and learning rates across phase change")
        # Refuse accidental overwrite, altered protocol, smoke-as-benchmark, and duplicate writer.
        try:
            train_model(data_root, work / "resumed", **kwargs)
            raise AssertionError("Overwrite was not rejected.")
        except FileExistsError:
            pass
        changed = TrainConfig(**{**asdict(config), "batch_size": 1})
        try:
            train_model(data_root, work / "resumed", **{**kwargs, "config": changed, "resume": True})
            raise AssertionError("Changed contract was not rejected.")
        except AssertionError as error:
            assert "恢复时" in str(error)
        try:
            evaluate_checkpoint(data_root, real_run, "val", device="cpu")
            raise AssertionError("Smoke checkpoint was accepted as a benchmark.")
        except AssertionError as error:
            assert "Smoke" in str(error)
        checks.append("Existing weights and changed resume configuration rejected; smoke cannot be reported as benchmark")
        write_json(ROOT / "reports" / "verification.json", {"status": "PASS", "checks": checks,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "real_model_check": real_check, "data": audit, "python": sys.executable,
            "torch": torch.__version__, "device_used": "cpu", "cuda_training_started": False,
            "full_training_completed": False, "gpu_runtime_verified": False,
            "internal_or_external_test_inference_run": False,
            "note": "Tiny training probe checks resume mechanics, not UGB accuracy. Real UGB checks use one CPU update and two validation images only."})
        print("ACCEPTANCE PASS: no benchmark result, no GPU speed or final accuracy claim.", flush=True)
    finally:
        # This directory is exclusively created by this script. No data/checkpoints outside it are removed.
        resolved = work.resolve()
        assert resolved.parent == ROOT.resolve() and resolved.name == "_qa_work" and not work.is_symlink()
        if work.exists():
            shutil.rmtree(resolved)


if __name__ == "__main__":
    main()
