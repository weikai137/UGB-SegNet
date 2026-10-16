"""Best-checkpoint inference, native/224 scoring, error panels and reviewable reports."""
import gc
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import ROOT, MANIFEST_SHA256, UPSTREAM_COMMIT, file_hash, manifest_rows, overlap_metrics, write_csv, write_json


def hd95_pixels(pred, truth):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    if not pred.any() or not truth.any():
        return 0. if not pred.any() and not truth.any() else None
    p = pred ^ binary_erosion(pred, border_value=0)
    t = truth ^ binary_erosion(truth, border_value=0)
    distances = np.concatenate([distance_transform_edt(~p)[t], distance_transform_edt(~t)[p]])
    return float(np.percentile(distances, 95))


def aggregate(scored):
    lesion = [row for row in scored if row["class"] != "normal"]
    normal = [row for row in scored if row["class"] == "normal"]
    def means(rows):
        result = {"n": len(rows)}
        for key in ("dice", "iou", "sensitivity", "specificity", "pixel_accuracy", "hd95_pixels"):
            values = [row[key] for row in rows if row[key] is not None]
            result[key] = float(np.mean(values)) if values else None
            result[key+"_undefined_n"] = len(rows)-len(values)
        return result
    return {"lesion_macro": means(lesion), "by_class": {
                label: means([row for row in lesion if row["class"] == label]) for label in ("benign", "malignant")},
            "normal_n": len(normal),
            "normal_false_positive_rate_any_pixel": sum(row["predicted_foreground_pixels"] > 0 for row in normal)/len(normal) if normal else None,
            "normal_foreground_pixels_mean": float(np.mean([row["predicted_foreground_pixels"] for row in normal])) if normal else None,
            "lesion_dice_median": float(np.median([row["dice"] for row in lesion])) if lesion else None,
            "lesion_dice_below_0.5_n": sum(row["dice"] < .5 for row in lesion),
            "lesion_empty_predictions_n": sum(row["predicted_foreground_pixels"] == 0 for row in lesion)}


def error_panel(data_root, row, prediction_path, output_path):
    with Image.open(Path(data_root) / row["image"]) as image:
        original = np.array(image.convert("RGB"))
    with Image.open(Path(data_root) / row["mask"]) as image:
        truth = np.array(image) > 0
    with Image.open(prediction_path) as image:
        pred = np.array(image) > 0
    assert truth.shape == pred.shape == original.shape[:2]
    error = original.astype(float)
    for region, color in ((pred & truth, [30, 205, 105]), (pred & ~truth, [255, 55, 55]), (~pred & truth, [40, 180, 255])):
        error[region] = .4*error[region] + .6*np.array(color)
    h, w = truth.shape
    from matplotlib import get_data_path
    font = ImageFont.truetype(str(Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"), 18)
    canvas = Image.new("RGB", (4*w+30, h+85), "#111827")
    draw = ImageDraw.Draw(canvas)
    metric = overlap_metrics(pred, truth)
    draw.text((5, 5), f"{row['id']}  Dice={metric['dice']:.4f}  recall={metric['sensitivity'] if metric['sensitivity'] is not None else 'N/A'}", fill="white", font=font)
    images = [Image.fromarray(original), Image.fromarray(truth.astype(np.uint8)*255).convert("RGB"),
              Image.fromarray(pred.astype(np.uint8)*255).convert("RGB"), Image.fromarray(error.astype(np.uint8))]
    for index, (image, title) in enumerate(zip(images, ["Original", "Ground truth", "Prediction", "TP green / FP red / FN blue"])):
        draw.text((index*(w+10), 35), title, fill="white", font=font)
        canvas.paste(image, (index*(w+10), 65))
    canvas.save(output_path)
    return metric


def evaluate_checkpoint(data_root, run_dir, split="val", device="cuda", *,
                        allow_smoke=False, rows_override=None, model_factory=None, output_base=None):
    import torch
    from ugb_adapter import BusiDataset, build_model
    assert split in {"val", "test", "external"}
    assert allow_smoke or (rows_override is None and model_factory is None and output_base is None)
    run_dir = Path(run_dir).resolve()
    checkpoint = run_dir / "checkpoint_best.pth"
    assert checkpoint.is_file(), "先训练到保存 best 权重，再运行本单元。"
    if split != "val" and not allow_smoke:
        assert (run_dir / "completed.json").is_file(), "完成训练后再评价测试集。"
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_base) / tag if output_base is not None else ROOT / "results" / run_dir.name / split / tag
    output.resolve().relative_to(ROOT.resolve())
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "evaluated_checkpoint.pth"
    before = checkpoint.stat()
    shutil.copy2(checkpoint, snapshot)
    after = checkpoint.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "权重在复制时更新，请重新执行。"
    snapshot_sha = file_hash(snapshot)
    state = torch.load(snapshot, map_location="cpu", weights_only=False)
    assert state["manifest_sha256"] == MANIFEST_SHA256
    assert state["upstream_commit"] == UPSTREAM_COMMIT
    if state["smoke"] and not allow_smoke:
        del state
        snapshot.unlink()
        output.rmdir()
        raise AssertionError("Smoke 权重不能作为正式结果。")
    device = torch.device(device)
    model = (model_factory or build_model)(pretrained=False).to(device).eval()
    model.load_state_dict(state["model"])
    rows = rows_override if rows_override is not None else manifest_rows(split)
    expected = {"val": 65, "test": 129, "external": 670}[split]
    assert allow_smoke or len(rows) == expected
    dataset = BusiDataset(rows, data_root)
    for name in ("predictions_native", "predictions_224", "uncertainty_native", "error_panels"):
        (output / name).mkdir()
    native_rows, resized_rows = [], []
    try:
        with torch.inference_mode():
            for index, row in enumerate(rows):
                image, target224 = dataset[index]
                result = model(image[None].to(device))
                probability224 = result["main"].float().sigmoid()
                probability_native = torch.nn.functional.interpolate(probability224,
                    size=(int(row["height"]), int(row["width"])), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                pred_native = probability_native > .5
                pred224 = probability224[0, 0].cpu().numpy() > .5
                with Image.open(Path(data_root) / row["mask"]) as image:
                    target_native = np.array(image) > 0
                assert pred_native.shape == target_native.shape == (int(row["height"]), int(row["width"]))
                for grid, prediction, truth, scored in (("native", pred_native, target_native, native_rows),
                                                        ("224", pred224, target224.numpy()[0] > .5, resized_rows)):
                    Image.fromarray(prediction.astype(np.uint8)).save(output / f"predictions_{grid}" / (row["id"]+".png"))
                    metric = overlap_metrics(prediction, truth)
                    metric["hd95_pixels"] = hd95_pixels(prediction, truth)
                    scored.append({"id": row["id"], "class": row["class"], **metric})
                if "uncertainty" in result:
                    uncertainty = torch.nn.functional.interpolate(result["uncertainty"].float(),
                        size=target_native.shape, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                    Image.fromarray(np.uint8(np.clip(uncertainty, 0, 1)*255)).save(output / "uncertainty_native" / (row["id"]+".png"))
                print(f"{split} [{index+1:03d}/{len(rows)}] {row['id']} native Dice={native_rows[-1]['dice']:.4f}", flush=True)
        ids = {row["id"] for row in rows}
        for grid in ("native", "224"):
            assert {p.stem for p in (output / f"predictions_{grid}").glob("*.png")} == ids
        row_map = {row["id"]: row for row in rows}
        ranked = sorted(native_rows, key=lambda row: row["dice"])
        # All validation/internal-test cases; a bounded gallery for the larger external set.
        chosen = ranked if len(rows) <= 129 else ranked[:20] + sorted((r for r in native_rows if r["class"] == "normal"), key=lambda r: r["predicted_foreground_pixels"], reverse=True)[:20]
        for row in chosen:
            error_panel(data_root, row_map[row["id"]], output / "predictions_native" / (row["id"]+".png"), output / "error_panels" / (row["id"]+".png"))
        write_csv(output / "native_per_image.csv", native_rows)
        write_csv(output / "224_per_image.csv", resized_rows)
        summary = {"status": "SMOKE_ONLY" if state["smoke"] else "COMPLETE", "split": split, "n": len(rows),
            "native": aggregate(native_rows), "224": aggregate(resized_rows),
            "manifest_sha256": MANIFEST_SHA256, "checkpoint_epoch": state["epoch"],
            "checkpoint_sha256": snapshot_sha, "checkpoint_snapshot": str(snapshot),
            "checkpoint_selection": "Best full-image macro validation Dice at 224x224, every 5 epochs by default; no EMA.",
            "checkpoint_selection_score_224": state["best_val_dice_224"],
            "smoke": state["smoke"], "upstream_commit": state["upstream_commit"],
            "native_grid": "Original image pixels: bilinear probability resize, then threshold >0.5.",
            "upstream_grid": "224x224 probabilities and nearest-resized truth, threshold >0.5.",
            "hd95": "Symmetric boundary-distance 95th percentile in pixels; one-sided empty masks undefined. Not the author's sampled-foreground implementation.",
            "uncertainty": "Encoder spatial-attention entropy from upstream output; not a calibrated lesion confidence.",
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "cautions": ["Image-group split; patient independence unverified.",
                "453/65/129 frozen split differs from the author's README 454/64/129.",
                "Validation, internal test and external results are separate; normal images excluded from lesion macro Dice.",
                "A validation score is not an independent test score; reported upstream results are not reproduced evidence."]}
        write_json(output / "summary.json", summary)
        write_evaluation_report(output, summary, ranked)
        print(f"完成：{output / 'report.html'}", flush=True)
        return output
    finally:
        del model, state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def write_evaluation_report(output, summary, ranked):
    cards = []
    for grid in ("native", "224"):
        metrics = summary[grid]["lesion_macro"]
        cards.append("<tr><td>"+grid+"</td>" + "".join(f"<td>{metrics[key]:.4f}</td>" if metrics[key] is not None else "<td>N/A</td>" for key in ("dice", "iou", "sensitivity", "specificity", "hd95_pixels")) + "</tr>")
    gallery = []
    for index, row in enumerate(ranked):
        panel = output / "error_panels" / (row["id"]+".png")
        if panel.exists():
            gallery.append(f'<details {"open" if index < 6 else ""}><summary>{row["id"]} · {row["class"]} · Dice {row["dice"]:.4f}</summary><img loading="lazy" src="error_panels/{row["id"]}.png"></details>')
    notes = "".join("<li>"+html.escape(note)+"</li>" for note in summary["cautions"])
    normal = summary["native"]
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UGB-SegNet {summary['split']} 评价</title><style>body{{font:16px/1.7 "Microsoft YaHei",sans-serif;background:#edf3f7;color:#173147;margin:0}}main{{max-width:1450px;margin:auto;padding:28px}}section,details{{background:white;border-radius:10px;padding:18px;margin:18px 0}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccd5dd;padding:9px}}img{{width:100%;height:auto}}summary{{cursor:pointer;font-weight:bold}}code{{overflow-wrap:anywhere}}</style><main><h1>UGB-SegNet · {summary['split']} · {summary['n']} 张</h1><p>状态 {summary['status']}；第 {summary['checkpoint_epoch']} 轮 best 权重。固定阈值 0.5，无后处理。</p><section><h2>逐图宏平均：两种网格分别报告</h2><table><tr><th>网格</th><th>Dice</th><th>IoU</th><th>敏感度</th><th>特异度</th><th>HD95 像素</th></tr>{''.join(cards)}</table><p>HD95 单侧空掩码未定义数：native {normal['lesion_macro']['hd95_pixels_undefined_n']}；224 {summary['224']['lesion_macro']['hd95_pixels_undefined_n']}。</p><p>正常图：{normal['normal_n']}；有任意误报像素的比例：{normal['normal_false_positive_rate_any_pixel']}。正常图不混入病灶 Dice。</p><p>native 适合与本工程 nnU-Net 原尺寸结果对照；224 与作者的重叠指标网格一致。HD95 采用真实边界距离，和作者随机抽取前景点的代码定义不同。</p></section><section><h2>口径与溯源</h2><ul>{notes}</ul><p>权重 SHA-256 <code>{summary['checkpoint_sha256']}</code></p><p>不确定图来自作者定义的注意力熵，不等于病灶置信概率。</p><p><a href="summary.json">完整汇总</a> · <a href="native_per_image.csv">原尺寸逐图 CSV</a> · <a href="224_per_image.csv">224 逐图 CSV</a></p></section><section><h2>原图 / 真值 / 预测 / 误分图（低 Dice 优先）</h2><p>绿：正确病灶 TP；红：多分 FP；蓝：漏分 FN。前六张默认展开，刻意展示最差情况。</p>{''.join(gallery)}</section></main></html>'''
    (output / "report.html").write_text(page, encoding="utf-8")
    metrics = summary["native"]["lesion_macro"]
    (output / "report.md").write_text(f"# UGB-SegNet {summary['split']}\n\n状态：{summary['status']}；{summary['n']} 张；best 第 {summary['checkpoint_epoch']} 轮。\n\n原尺寸逐图病灶宏平均：\n\n" + "\n".join(f"- {key}: {metrics[key]}" for key in ("dice", "iou", "sensitivity", "specificity", "hd95_pixels")) + "\n\n[完整误分图库](report.html)\n\n" + "\n".join("- "+note for note in summary["cautions"]), encoding="utf-8")
