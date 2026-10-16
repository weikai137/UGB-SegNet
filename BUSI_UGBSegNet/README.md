# BUSI UGB-SegNet：手动 Notebook 复刻

日常入口是 **BUSI_UGBSegNet.ipynb**，使用共享 **conda cv**。按顺序逐格运行，与现有 nnU-Net 的操作方式一致；不自动开始训练，不启动后台任务，也不加入自动曲线刷新。

## 从 GitHub 获取和更新代码

```bash
git clone https://github.com/weikai137/UGB-SegNet.git
cd UGB-SegNet/BUSI_UGBSegNet
bash Setup-AutoDL.sh
```

GitHub 保存代码、环境配置、固定划分和来源记录；不上传数据图片、训练权重、运行报告或 ZIP。缺少 ImageNet 骨干时，安装脚本会从官方地址下载约 21 MB 并核对完整 SHA-256，已有权重直接复用。

数据只需上传一次。若已在仓库旁解压完整 AutoDL 包，例如 `/root/autodl-tmp/BUSI_UGBSegNet/data/prepared`，仓库版会自动复用；也可在 Notebook 的 `DATA_ROOT_OVERRIDE` 或环境变量 `BUSI_DATA_ROOT` 中指定准备数据的根目录。该目录下应有 `nnUNet_raw/` 和 `data/BUS_UCLM/`。

以后在仓库内运行 `git pull --ff-only` 更新代码，运行目录与数据留在本机。GitHub 版 Notebook 清除了执行输出和嵌入图片；自己运行后、再次提交前，可执行 `python -B scripts/clear_notebook_outputs.py` 清除 Notebook 输出，磁盘训练日志和权重保留。

## AutoDL 开始运行

1. 选择支持 CUDA 的 GPU 实例，使用包含 conda 的 Linux 镜像，优先使用 CUDA 12.8 / PyTorch 2.7 系列镜像。
2. 上传 `BUSI_UGBSegNet_AutoDL.zip` 到实例数据盘目录，例如 `/root/autodl-tmp`，解压：

   ```bash
   cd /root/autodl-tmp
   unzip BUSI_UGBSegNet_AutoDL.zip
   cd BUSI_UGBSegNet
   bash Setup-AutoDL.sh
   ```

3. 在本地 VS Code 使用 Remote-SSH 连接实例。远程窗口安装 Python / Jupyter 扩展，打开远程工程的 `BUSI_UGBSegNet.ipynb`，选择 **Python (conda cv)** 内核。
4. 逐格运行第 1–5 节准备代码。首次 `RESUME=False`，运行第 6 节才开始训练。
5. 训练正常结束或早停后，手动运行第 8 节全部验证、第 9 节内部测试、第 10 节外部测试，再查看误分图和报告。

远程连接步骤参考 [AutoDL 官方 VS Code 文档](https://www.autodl.com/docs/vscode/)。实例必须保持开机；若连接中断，重连后先看日志和运行状态，已有训练仍在运行时不要重复启动。需要恢复时将 `RESUME=True`，重新执行准备与训练单元。

环境只在新机器首次安装。已安装的匹配版 torch / torchvision 会复用。安装脚本使用 conda 创建或复用 cv，再通过该环境的 pip 安装依赖，关闭安装包缓存，最后检查 CUDA 和依赖一致性。配置使用 [PyTorch 官方提供的 2.7.1 / CUDA 12.8 安装组合](https://pytorch.org/get-started/previous-versions/#v2-7-1)，不照旧论文硬装 CUDA 11.7。


镜像说明：平台显示的 CUDA 版本可能是驱动支持的上限；本工程固定安装 PyTorch 2.7.1 + CUDA 12.8。新驱动可向下兼容 CUDA 12.x 应用，无需因为平台显示 CUDA 13.0 就改装另一套 PyTorch。安装完成后，以脚本的 CUDA 可用性检查和 Notebook 环境检查为准。参考 [NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

## 默认实验

| 项目 | 设置 |
| --- | --- |
| 模型 | 作者当前 `UGBSegNetImproved`，源码不改 |
| 上游版本 | `20c8e4ea20691a49dd07c65cd1e3f44ef3eff2a4` |
| 参数量，代码实测 | **5,043,693**；与作者 README 的 14.00M 不一致 |
| 训练 / 验证 / 内部测试 | **453 / 65 / 129**，与 nnU-Net 共用同一份固定划分 |
| 外部测试 | BUS-UCLM 670 张：病灶 260、正常 410；沿用原协议排除 13 张几何异常样本 |
| 输入 | 224×224，灰度复制为 3 通道，ImageNet 标准化 |
| 预训练 | EfficientNet-B0 IMAGENET1K_V1；交付包已包含官方权重与哈希 |
| batch / 上限 | **16 / 120 轮** |
| 第一阶段 | 第 1–50 轮冻结骨干阶段 0–3 的参数，AdamW，lr=1e-4 |
| 第二阶段 | 第 51 轮起全部解冻，重建 AdamW，lr=3e-5 |
| 调度 | 作者的 CosineAnnealingWarmRestarts；阶段 1 T0=120、阶段 2 T0=70，eta_min=1e-6 |
| 损失 | 作者 SegLoss：Dice + Focal + 5×boundary + 3 个深监督项（各 0.4） |
| best / 早停 | 每 5 轮检查完整 224 验证 Dice；连续 25 轮未改善早停；不使用 EMA |
| 运行精度 | 默认 FP32，AMP=False；保留官方梯度检查点机制 |
| 保存 | latest 每轮；best 在选取点改善时；final 在正常结束或早停时 |

每轮都输出训练损失、验证损失、全部 65 张的 224 Dice、学习率和耗时。额外的逐轮验证仅供观察，默认 best / 早停仍按每 5 轮检查，保持作者训练驱动的选取频率。

**120 是训练上限，不保证一定执行到 120。** 作者早停逻辑可能在全部解冻之前触发。本工程不擅自修改这个行为。要修改训练预算、batch、AMP 或选取规则时，请使用新的 RUN_NAME，不能当作原实验无差别续跑。

## 与源码的适配与区别

依据 [作者仓库](https://github.com/taukiralam007/UGB-SegNet/tree/20c8e4ea20691a49dd07c65cd1e3f44ef3eff2a4)，保留模型、损失和两阶段训练方案。工程适配明确如下：

- 将五折训练驱动改为与 nnU-Net 对齐的单次固定划分；作者 README 写 454/64/129，不能据此宣称两者逐图相同。
- 准备数据为 0/1 掩码，读取使用 `>0`，修复直接沿用作者 `>127` 时会丢失全部前景的问题。多个病灶掩码已取并集。
- 输入来自现有灰度准备数据，复制为 RGB，而不是重新读取可能带彩色标记的原图。
- 使用源码同类增强且强制成对变换图片和掩码。缺少 albumentations 会报错，不走作者仅改变图片而未同步改变掩码的降级分支。固定 albumentations 1.4.24，避免不同大版本默认行为漂移。
- 增加逐轮日志、原子断点、优化器 / 调度器 / RNG 恢复、单写入锁和运行配置核对。CPU 检查已验证冻结与解冻两个恢复点的连续性；CUDA 不作位级重放保证。
- 默认 FP32 与原代码一致；AMP 是可选变体。额外的验证损失与每轮 Dice 只用于观察。

这是**官方当前代码的固定划分复刻工程**，并非作者五折结果的独立重现。未执行完整训练前，不声称获得 README 的 Dice 81.30%。

## 评价口径

每次评价同时导出两套结果：

1. **native 原尺寸**：224 的概率图双线性还原到原图尺寸，再以 `>0.5` 得到掩码，与原图真值逐图比较。该结果可与本工程 nnU-Net 的原尺寸评价对照。
2. **224×224**：概率图和最近邻缩放的真值直接比较。Dice / IoU / 敏感度 / 特异度的网格与作者代码一致，但数据划分不同。

两套均导出 Dice、IoU、敏感度、特异度、辅助像素准确率、边界 HD95 和逐图 CSV。HD95 使用真实边界距离，单侧空掩码记未定义并统计数量；**不同于作者随机取最多 300 个前景点且空掩码返回 0 的实现**，不要直接比较它们的 HD95。距离单位为像素，不是毫米。

验证集、内部测试、外部测试单独报告。正常图不混入病灶 Dice，另报有任意误报像素的比例。误分图中绿为 TP、红为 FP、蓝为 FN；源码的不确定图属于编码器注意力熵，不是校准过的病灶置信概率。

## 文件位置

- `BUSI_UGBSegNet.ipynb`：唯一日常入口，包含可阅读的完整训练循环。
- `runs/ugb_seed42/`：模型权重、history.csv、progress.png、训练配置及结束状态。
- `results/ugb_seed42/<split>/<时间>/`：每次独立评价的固定权重快照、原尺寸 / 224 预测、逐图指标、误分图和 HTML / Markdown 报告。
- `data/manifest.csv`、`splits_final.json`：与 nnU-Net 相同的冻结数据协议。
- `data/prepared/`：仅完整上传包内携带，不在本机重复复制数据。
- `vendor/UGB-SegNet/`：原始代码和 README，附 commit 与文件哈希。
- `pretrained/`：必要的 ImageNet 初始化权重，属于运行资产，不是临时缓存。
- `reports/verification.json`：真实 CPU 检查记录；不是完整训练或 GPU 性能成绩。

本地数据自动读取旁边 `BUSI_nnUNet` 的准备数据，不改动原工程。上传包独立携带全部 1317 对准备数据，不需要上传 nnU-Net 权重、预处理数组或旧环境。

需要重新打包时，可手动运行 `python -B scripts/package_autodl.py --output <新的zip路径>`，已有 ZIP 不会被覆盖。
