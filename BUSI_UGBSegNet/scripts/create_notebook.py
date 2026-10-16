"""Generate the manual notebook; training code is visible in the notebook itself."""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def md(text): cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text, safe=False):
        cell = nbformat.v4.new_code_cell(text)
        cell.metadata["manual_compute"] = not safe
        cells.append(cell)
    md('''# BUSI UGB-SegNet：AutoDL 手动复刻

统一使用 **conda cv / Python 3.12 / PyTorch 2.7.1 CUDA 12.8**，右上角选择 **Python (conda cv)**。
依次逐格运行：环境 → 参数 → 数据检查 → 样本 → 模型 → 训练代码 → 手动训练 → 曲线 → 完整验证 → 两套测试 → 误分图与报告。

**首次 RESUME=False；中断后 RESUME=True。没有自动启动或后台训练。** 曲线保存到磁盘，下方单元可手动查看，不加入后台刷新。

模型、损失、120 轮上限、前 50 轮部分冻结 / 第 51 轮全部解冻、AdamW 与余弦调度来自[作者代码](https://github.com/taukiralam007/UGB-SegNet/tree/20c8e4ea20691a49dd07c65cd1e3f44ef3eff2a4)。
沿用 nnU-Net 固定的 **453 / 65 / 129** 划分；不是作者 README 的 454 / 64 / 129，也不冒充作者五折结果。
''')
    md('''## 1. 环境与路径
本地直接复用旁边 BUSI_nnUNet 的准备数据；AutoDL 完整 ZIP 自带数据和 ImageNet 骨干权重，无需重新划分或在线下载预训练权重。新机器先在终端运行 `bash Setup-AutoDL.sh`，日常无需重复安装。''')
    code('''import os, sys, json, gc, csv, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from IPython.display import display, Markdown, Image as NotebookImage
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
ROOT = Path.cwd().resolve()
if not (ROOT / "vendor" / "provenance.json").exists(): ROOT = ROOT / "BUSI_UGBSegNet"
assert (ROOT / "vendor" / "provenance.json").exists(), "请从 BUSI_UGBSegNet 工程目录打开 Notebook。"
assert (Path(sys.prefix) / "conda-meta").is_dir(), "请选择 conda cv 内核。"
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
from common import find_data_root, manifest_rows, verify_data, write_json, MANIFEST_SHA256, UPSTREAM_COMMIT
from ugb_adapter import build_model, build_loss, BusiDataset
VERIFY_ONLY = os.environ.get("UGB_VERIFY_ONLY") == "1"
gpu = "CPU 只读验证；未验证 GPU 完整训练" if VERIFY_ONLY else ("CUDA GPU" if torch.cuda.is_available() else "当前没有 CUDA GPU")
display(pd.DataFrame({"项目": ["Python", "Conda", "PyTorch", "CUDA runtime", "GPU", "工程", "上游版本"],
                      "值": [sys.executable, sys.prefix, torch.__version__, torch.version.cuda, gpu, str(ROOT), UPSTREAM_COMMIT]}))
''', True)
    md('''## 2. 参数与恢复
默认采用作者训练方案。**120 是上限；连续 25 轮没有改善时可提前停止**，best 与早停按每 5 轮的完整 224 验证 Dice 决定。每轮都额外打印验证结果供观察。

`AMP=False` 保留源码的 FP32；若以后设 True，属于显式的混合精度变体，应使用新的 RUN_NAME。当前选择规则没有 EMA。''')
    code('''RUN_NAME = "ugb_seed42"
RESUME = False
DATA_ROOT_OVERRIDE = None  # 自定义数据目录时填写；通常无需修改
DEVICE = "cuda"
CONFIG = dict(epochs=120, batch_size=16, workers=0 if os.name == "nt" else 4,
              cpu_threads=4, phase1_end=50, lr_phase1=1e-4, lr_phase2=3e-5,
              weight_decay=1e-5, eta_min=1e-6, selection_every=5, patience=25,
              seed=42, image_size=224, threshold=.5, amp=False, save_every=1)
assert RUN_NAME.replace("_", "").replace("-", "").isalnum()
RUN_DIR = ROOT / "runs" / RUN_NAME
DATA_ROOT = find_data_root(DATA_ROOT_OVERRIDE)
manifest = pd.DataFrame(manifest_rows())
display(pd.DataFrame({"参数": list(CONFIG), "值": [str(v) for v in CONFIG.values()]}))
print("数据：", DATA_ROOT, "；输出：", RUN_DIR, "；恢复：", RESUME)
''', True)
    md('''## 3. 固定数据检查
逐对检查原尺寸、0/1 标签和像素哈希。所有多掩码已合并；完全相同的图像留在同一划分，病人级独立性尚未核实。此处只做完整性检查，不进行测试集性能评价。''')
    code('''audit = verify_data(DATA_ROOT, deep=True)
write_json(ROOT / "reports" / "data_check.json", audit)
display(manifest.groupby(["dataset", "split", "class"]).size().unstack(fill_value=0))
print("PASS：", audit["pairs"], "对；manifest SHA-256：", MANIFEST_SHA256)
''', True)
    md('''### 看训练样本
固定取良性和恶性各两张；绿色是真值。网络读取时按作者方式缩到 224×224，灰度图复制为 RGB，再做 ImageNet 标准化。掩码始终使用最近邻插值，不能按图片平滑插值。''')
    code('''import matplotlib.pyplot as plt
examples = pd.concat([manifest[(manifest["split"] == "train") & (manifest["class"] == label)].sort_values("id").head(2)
                      for label in ["benign", "malignant"]])
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for col, (_, row) in enumerate(examples.iterrows()):
    with Image.open(DATA_ROOT / row["image"]) as image: pixels = np.array(image)
    with Image.open(DATA_ROOT / row["mask"]) as image: mask = np.array(image) > 0
    axes[0, col].imshow(pixels, cmap="gray"); axes[0, col].set_title(row["id"], fontsize=9)
    axes[1, col].imshow(pixels, cmap="gray")
    axes[1, col].imshow(np.ma.masked_where(~mask, mask), cmap="Greens", alpha=.55, vmin=0, vmax=1)
    for ax in axes[:, col]: ax.axis("off")
plt.tight_layout(); plt.show()
''', True)
    md('''## 4. 模型和损失
直接加载固定版本的 `UGBSegNetImproved` 和原始 `SegLoss`，不重新实现网络。EfficientNet-B0 + 三尺度 CBAM + LearnFPN + 不确定性解码器 + 三个深监督头。此单元只在 CPU 建模检查，不开始训练。''')
    code('''preview_model = build_model(pretrained=True)
preview_loss = build_loss()
print("模型：", type(preview_model).__name__)
print("参数量：", f"{sum(p.numel() for p in preview_model.parameters()):,}")
print("损失：", preview_loss)
print("骨干权重：本工程 pretrained/，已核对 SHA-256。")
del preview_model, preview_loss
gc.collect()
''')
    md('''## 5. 完整训练代码
下面包含训练循环、验证、两阶段优化器、原子保存和断点恢复，可直接阅读。定义函数不会启动训练。latest 每轮保存；best 在每 5 轮的选取点改善时保存；final 在正常结束或早停后保存。''')
    code((ROOT / "scripts" / "training.py").read_text(encoding="utf-8"), True)
    md('''## 6. 手动开始或恢复训练
**运行这一格才会开始长时间训练。** 每轮打印训练损失、验证损失、全部 65 张的 224 Dice、学习率和耗时。这里的验证没有随机 patch 采样，也没有 EMA。

中断后将 `RESUME=True`，重新执行准备单元和本格；恢复已保存的完整轮次。改变 batch、AMP、总轮数或选取规则时另起 RUN_NAME。''')
    code('''assert torch.cuda.is_available(), "请先在 GPU 实例上准备 CUDA 环境。"
training_result = train_model(DATA_ROOT, RUN_DIR, TrainConfig(**CONFIG), resume=RESUME, device=DEVICE)
display(training_result)
''')
    md('''## 7. 手动查看曲线和断点
训练每轮保存 progress.png。训练暂停或结束后运行此格查看；运行中的训练占用内核，后面的单元会排队。''')
    code('''curve_path = RUN_DIR / "progress.png"
if curve_path.exists(): display(NotebookImage(filename=str(curve_path), width=1100))
else: print("尚无已完成轮次的训练曲线。")
history_path = RUN_DIR / "history.csv"
if history_path.exists(): display(pd.read_csv(history_path).tail(10))
display(pd.DataFrame([{"权重": path.name, "MB": round(path.stat().st_size / 2**20, 1)}
                      for path in RUN_DIR.glob("checkpoint_*.pth")]))
''', True)
    md('''## 8. 手动评价全部 65 张验证图
读取 best 的固定副本。同时给出原尺寸指标和 224×224 指标，逐图计算 Dice、IoU、敏感度、特异度、像素准确率、边界 HD95。固定阈值 0.5，无后处理。

原尺寸指标可与 nnU-Net 对照；输入仍是 224，概率图双线性恢复原尺寸后再阈值。此单元可在训练暂停后做中途检查。''')
    code('''from evaluation import evaluate_checkpoint
VAL_OUTPUT = evaluate_checkpoint(DATA_ROOT, RUN_DIR, "val", device=DEVICE)
val_summary = json.loads((VAL_OUTPUT / "summary.json").read_text())
display(pd.DataFrame({grid: val_summary[grid]["lesion_macro"] for grid in ["native", "224"]}))
display(Markdown(f"[打开验证报告和误分图库]({VAL_OUTPUT.relative_to(ROOT).as_posix()}/report.html)"))
''')
    md('''## 9. 手动内部测试：129 张
完成训练或早停后再执行。测试图不参与 best 选择。结果单独保存，不与验证结果混算。''')
    code('''TEST_OUTPUT = evaluate_checkpoint(DATA_ROOT, RUN_DIR, "test", device=DEVICE)
test_summary = json.loads((TEST_OUTPUT / "summary.json").read_text())
display(pd.DataFrame({grid: test_summary[grid]["lesion_macro"] for grid in ["native", "224"]}))
display(Markdown(f"[打开内部测试报告]({TEST_OUTPUT.relative_to(ROOT).as_posix()}/report.html)"))
''')
    md('''## 10. 手动外部测试：BUS-UCLM 670 张
病灶 260 张和正常图 410 张分别报告：正常图单独统计误报，不混进病灶 Dice。原始 13 张几何不匹配样本已按固定数据协议排除。外部集不参与训练或阈值调整。''')
    code('''EXTERNAL_OUTPUT = evaluate_checkpoint(DATA_ROOT, RUN_DIR, "external", device=DEVICE)
external_summary = json.loads((EXTERNAL_OUTPUT / "summary.json").read_text())
display(pd.DataFrame({grid: external_summary[grid]["lesion_macro"] for grid in ["native", "224"]}))
display({key: external_summary["native"][key] for key in ["normal_n", "normal_false_positive_rate_any_pixel", "normal_foreground_pixels_mean"]})
display(Markdown(f"[打开外部测试报告]({EXTERNAL_OUTPUT.relative_to(ROOT).as_posix()}/report.html)"))
''')
    md('''## 11. 选一张验证图看误分和注意力熵
设置 CASE_ID；不设置则读取验证集最低 Dice 的一张。绿色 TP、红色 FP、蓝色 FN。熵图来自源码的注意力熵，不是校准过的病灶概率。''')
    code('''from evaluation import error_panel
CASE_ID = None
assert "VAL_OUTPUT" in globals(), "先执行第 8 节完整验证。"
scores = pd.read_csv(VAL_OUTPUT / "native_per_image.csv").sort_values("dice")
display(scores[["id", "class", "dice", "sensitivity", "fp", "fn"]].head(10))
case_id = CASE_ID or scores.iloc[0]["id"]
row = manifest[manifest["id"] == case_id].iloc[0].to_dict()
panel = VAL_OUTPUT / "error_panels" / f"{case_id}.png"
if not panel.exists(): error_panel(DATA_ROOT, row, VAL_OUTPUT / "predictions_native" / f"{case_id}.png", panel)
display(NotebookImage(filename=str(panel), width=1400))
uncertainty = VAL_OUTPUT / "uncertainty_native" / f"{case_id}.png"
if uncertainty.exists(): display(NotebookImage(filename=str(uncertainty), width=550))
''')
    md('''## 12. 查看全部结果报告
每次评价自动写入 HTML、Markdown、JSON、逐图 CSV、原尺寸预测和误分图。下面列出已有报告，尚未运行的项目不会显示成已完成。''')
    code('''reports = sorted((ROOT / "results" / RUN_NAME).glob("*/*/report.html"))
for report in reports:
    display(Markdown(f"[{report.parent.parent.name} · {report.parent.name}]({report.relative_to(ROOT).as_posix()})"))
if not reports: print("尚未评价；训练完成后按顺序手动运行验证、内部测试和外部测试单元。")
''', True)
    md('''## 13. 使用边界与复现说明

- 这是**官方模型与损失的固定划分复刻工程**；作者发布的是五折训练驱动，本工程为了和 nnU-Net 同图比较使用单次固定划分。
- 数据读取修复：0/1 掩码用 `>0`；强制成对增强，缺少 albumentations 会报错，不启用作者未同步变换掩码的降级分支；全部多掩码已合并。
- 每轮额外验证用于观察；best 与早停默认仍每 5 轮检查。前 50 轮冻结阶段 0–3 的参数，BatchNorm 行为保留原源码。
- 224 下的 Dice / IoU 可核对作者计算网格；HD95 使用真实边界距离，单侧空掩码记未定义，区别于作者的随机前景点实现。没有毫米间距信息。
- 未运行完整 GPU 训练前，不声称复现了 README 的 81.30%。CPU 检查只证明实现可运行，不代表最终精度或远程吞吐。
- 停止按钮会中断当前单元。再次训练先确认 RESUME 和 RUN_NAME；不要同时启动两份同名实验。
''')
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.update(kernelspec={"display_name": "Python (conda cv)", "language": "python", "name": "cv"},
        language_info={"name": "python", "version": "3.12"})
    nbformat.validate(nb)
    nbformat.write(nb, ROOT / "BUSI_UGBSegNet.ipynb")
    print(f"Notebook created: {len(cells)} cells, {sum(c.cell_type == 'code' for c in cells)} code cells.")


if __name__ == "__main__":
    main()
