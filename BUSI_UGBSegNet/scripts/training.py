"""Manual single-split training using the pinned UGB-SegNet architecture and loss."""
import gc
import json
import math
import random
import time
import os
from functools import wraps
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import ROOT, MANIFEST_SHA256, UPSTREAM_COMMIT, manifest_rows, overlap_metrics, write_csv, write_json
from ugb_adapter import BusiDataset, build_loss, build_model, seed_worker


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 16
    workers: int = 4
    cpu_threads: int = 4
    phase1_end: int = 50
    lr_phase1: float = 1e-4
    lr_phase2: float = 3e-5
    weight_decay: float = 1e-5
    eta_min: float = 1e-6
    selection_every: int = 5
    patience: int = 25
    seed: int = 42
    image_size: int = 224
    threshold: float = .5
    amp: bool = False
    save_every: int = 1

    def contract(self):
        # Worker/thread counts are runtime choices, recorded but portable.
        return {k: v for k, v in asdict(self).items() if k not in ("workers", "cpu_threads", "save_every")}


def make_optimizer(model, config, phase):
    lr = config.lr_phase1 if phase == 1 else config.lr_phase2
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
        T_0=config.epochs if phase == 1 else config.epochs-config.phase1_end,
        T_mult=1, eta_min=config.eta_min)
    return optimizer, scheduler


def atomic_torch_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.inference_mode()
def validation_pass(model, criterion, loader, device, amp=False):
    model.eval()
    losses, scored = [], []
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            loss = criterion(output, targets, is_ugbnet=True)
        assert torch.isfinite(loss), "Non-finite validation loss."
        losses.append(float(loss))
        probabilities = output["main"].float().sigmoid().cpu().numpy()[:, 0]
        for prob, truth in zip(probabilities, targets.cpu().numpy()[:, 0]):
            scored.append(overlap_metrics(prob > .5, truth > .5))
    assert scored
    return {"val_loss": float(np.mean(losses)), **{
        "val_"+key+"_224": float(np.mean([row[key] for row in scored]))
        for key in ("dice", "iou", "sensitivity", "specificity")}}


def save_training_curve(history, run_dir):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    epochs = [row["epoch"] for row in history]
    fig = Figure(figsize=(12, 8))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    for key, label in (("train_loss", "train"), ("val_loss", "validation")):
        axes[0, 0].plot(epochs, [row[key] for row in history], label=label)
    axes[0, 0].set_title("Loss"); axes[0, 0].legend()
    axes[0, 1].plot(epochs, [row["val_dice_224"] for row in history], color="#0b8575")
    selected = [row for row in history if row["selection_check"]]
    axes[0, 1].scatter([row["epoch"] for row in selected], [row["val_dice_224"] for row in selected], s=18, label="selection check")
    axes[0, 1].set(title="Full validation at 224 x 224 (no EMA)", ylim=(0, 1)); axes[0, 1].legend()
    axes[1, 0].plot(epochs, [row["seconds"] for row in history]); axes[1, 0].set(title="Epoch duration", ylabel="seconds")
    axes[1, 1].plot(epochs, [row["lr_used"] for row in history]); axes[1, 1].set(title="Learning rate used", ylabel="learning rate")
    for ax in axes.flat:
        ax.set_xlabel("Completed epoch (1-based)"); ax.grid(alpha=.2)
    fig.tight_layout()
    temporary = Path(run_dir) / "progress.tmp.png"
    fig.savefig(temporary, dpi=130)
    temporary.replace(Path(run_dir) / "progress.png")
    fig.clear()


def exclusive_training(function):
    @wraps(function)
    def locked(data_root, run_dir, *args, **kwargs):
        directory = Path(run_dir).resolve()
        directory.relative_to(ROOT.resolve())
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".training.lock").open("a+b") as lock:
            if lock.tell() == 0:
                lock.write(b"0"); lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return function(data_root, run_dir, *args, **kwargs)
    return locked


@exclusive_training
def train_model(data_root, run_dir, config=None, resume=False, device="cuda", *,
                smoke=False, stop_after_epoch=None, model_factory=None, rows_override=None):
    config = config or TrainConfig()
    assert config.epochs > config.phase1_end >= 1
    assert config.batch_size >= 1 and config.workers >= 0
    assert config.selection_every >= 1 and config.save_every >= 1
    assert config.image_size == 224 and config.threshold == .5
    assert config.patience >= 1
    assert not rows_override or smoke, "Partial data are allowed only in an explicitly marked smoke check."
    assert model_factory is None or smoke, "Production training must use the pinned model."
    run_dir = Path(run_dir).resolve()
    run_dir.relative_to(ROOT.resolve())
    run_dir.mkdir(parents=True, exist_ok=True)
    latest, final = run_dir / "checkpoint_latest.pth", run_dir / "checkpoint_final.pth"
    if not resume and any(run_dir.glob("checkpoint_*.pth")):
        raise FileExistsError("已有权重：设 RESUME=True，或使用新的 RUN_NAME。")
    if resume and not latest.exists():
        if final.exists() and (run_dir / "completed.json").exists():
            print("此实验已结束；可运行验证、测试或报告单元。")
            return json.loads((run_dir / "completed.json").read_text())
        raise FileNotFoundError("没有 latest 断点。首次训练请设 RESUME=False。")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前内核不能使用 CUDA；请在 AutoDL 选择 conda cv 内核。")
    torch.set_num_threads(config.cpu_threads)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False) if resume else None
    if checkpoint:
        assert checkpoint["manifest_sha256"] == MANIFEST_SHA256
        assert checkpoint["upstream_commit"] == UPSTREAM_COMMIT
        assert checkpoint["smoke"] == smoke
        assert checkpoint["contract"] == config.contract(), "恢复时须保持训练计划、batch、精度和模型选择规则一致。"
        if (run_dir / "completed.json").is_file():
            completed = json.loads((run_dir / "completed.json").read_text())
            assert completed["completed_epochs"] == checkpoint["epoch"], "结束记录与 latest 不一致，请先检查该实验目录。"
            print("此实验已正常结束或早停；继续查看评价结果即可。需要新实验请换 RUN_NAME。", flush=True)
            return completed
        if checkpoint["epoch"] >= config.epochs:
            atomic_torch_save(checkpoint, final)
            result = {"status": "TRAINING_COMPLETE", "completed_epochs": checkpoint["epoch"],
                "planned_epochs": config.epochs, "best_epoch": checkpoint["best_epoch"],
                "best_val_dice_224": checkpoint["best_val_dice_224"], "smoke": smoke,
                "manifest_sha256": MANIFEST_SHA256, "upstream_commit": UPSTREAM_COMMIT}
            write_json(run_dir / "completed.json", result)
            return result
    factory = model_factory or build_model
    model = factory(pretrained=not resume).to(device)
    phase = checkpoint["phase"] if checkpoint else 1
    if phase == 1:
        model.freeze_backbone(n=4)
    else:
        model.unfreeze_all()
    optimizer, scheduler = make_optimizer(model, config, phase)
    criterion = build_loss().to(device)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, best, best_epoch, stale, start = [], -math.inf, None, 0, 1
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        history, best, best_epoch, stale = checkpoint["history"], checkpoint["best_val_dice_224"], checkpoint["best_epoch"], checkpoint["stale"]
        start = checkpoint["epoch"] + 1
        restore_rng(checkpoint["rng"], device)
        write_csv(run_dir / "history.csv", history)
        del checkpoint
    train_rows, val_rows = rows_override or (manifest_rows("train"), manifest_rows("val"))
    assert smoke or (len(train_rows) == 453 and len(val_rows) == 65)
    train_set = BusiDataset(train_rows, data_root, augment=True, seed=config.seed)
    val_set = BusiDataset(val_rows, data_root)
    loader_args = {"num_workers": config.workers, "pin_memory": device.type == "cuda"}
    if config.workers:
        loader_args.update(multiprocessing_context="spawn", worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, **loader_args)
    write_json(run_dir / "config.json", {"config": asdict(config), "manifest_sha256": MANIFEST_SHA256,
        "upstream_commit": UPSTREAM_COMMIT, "smoke": smoke, "device": str(device),
        "selection": "macro Dice of every validation image at 224x224, every 5 epochs by default",
        "validation_logged_every_epoch": True, "data_root": str(data_root),
        "resume": "Full completed epochs only; augmentation is seeded per epoch. CUDA bitwise replay is not guaranteed."})
    early_stopped = False
    print(f"UGB-SegNet | train={len(train_rows)} val={len(val_rows)} | epochs {start}..{config.epochs} | {device} | AMP={use_amp}", flush=True)
    last_state = None
    try:
        for epoch in range(start, config.epochs + 1):
            if epoch == config.phase1_end+1 and phase == 1:
                phase = 2
                model.unfreeze_all()
                optimizer, scheduler = make_optimizer(model, config, phase)
                print(f"Phase 2: all layers unfrozen; lr={config.lr_phase2:g}", flush=True)
            tick = time.perf_counter()
            model.train()
            train_set.set_epoch(epoch)
            generator = torch.Generator().manual_seed(config.seed+epoch*1009)
            loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                                generator=generator, **loader_args)
            losses = []
            lr_used = optimizer.param_groups[0]["lr"]
            for images, targets in loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    loss = criterion(model(images), targets, is_ugbnet=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"Non-finite gradient at epoch {epoch}.")
                scaler.step(optimizer); scaler.update()
                losses.append(float(loss.detach()))
            scheduler.step()
            validation = validation_pass(model, criterion, val_loader, device, use_amp)
            selection = epoch % config.selection_every == 0 or epoch == config.epochs
            improved = selection and validation["val_dice_224"] > best
            if selection:
                if improved:
                    best, best_epoch, stale = validation["val_dice_224"], epoch, 0
                else:
                    stale += config.selection_every
            record = {"epoch": epoch, "phase": phase, "train_loss": float(np.mean(losses)),
                **validation, "lr_used": lr_used, "lr_next": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter()-tick, "selection_check": selection,
                "best_updated": improved, "best_val_dice_224": best if math.isfinite(best) else None}
            history.append(record)
            last_state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(device),
                "epoch": epoch, "phase": phase, "best_val_dice_224": best, "best_epoch": best_epoch,
                "stale": stale, "history": history, "contract": config.contract(),
                "manifest_sha256": MANIFEST_SHA256, "upstream_commit": UPSTREAM_COMMIT, "smoke": smoke}
            if epoch % config.save_every == 0 or epoch == config.epochs:
                atomic_torch_save(last_state, latest)
            if improved:
                atomic_torch_save({k: last_state[k] for k in ("model", "epoch", "phase", "best_val_dice_224", "best_epoch", "contract", "manifest_sha256", "upstream_commit", "smoke")}, run_dir / "checkpoint_best.pth")
            write_csv(run_dir / "history.csv", history)
            save_training_curve(history, run_dir)
            print(f"Epoch {epoch:03d}/{config.epochs} | phase {phase} | loss {record['train_loss']:.4f} | val_loss {record['val_loss']:.4f} | val Dice(224) {record['val_dice_224']:.4f} | lr {lr_used:.7f} | {record['seconds']:.1f}s" + (" | NEW BEST" if improved else ""), flush=True)
            if selection and stale >= config.patience:
                early_stopped = True
                break
            if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < config.epochs:
                atomic_torch_save(last_state, latest)
                print("验证用暂停：断点已保存，训练计划未改变。", flush=True)
                return {"status": "CHECKPOINT_SAVED", "epoch": epoch, "smoke": smoke}
        assert last_state is not None
        atomic_torch_save(last_state, final)
        atomic_torch_save(last_state, latest)
        result = {"status": "EARLY_STOPPED" if early_stopped else "TRAINING_COMPLETE",
            "completed_epochs": history[-1]["epoch"], "planned_epochs": config.epochs,
            "best_epoch": best_epoch, "best_val_dice_224": best, "smoke": smoke,
            "manifest_sha256": MANIFEST_SHA256, "upstream_commit": UPSTREAM_COMMIT}
        write_json(run_dir / "completed.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    except KeyboardInterrupt:
        print("训练已中断；恢复使用 latest 中已保存的完整轮次。", flush=True)
        raise
    finally:
        del model, optimizer, scheduler, criterion, last_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
