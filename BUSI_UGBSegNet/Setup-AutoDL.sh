#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v conda >/dev/null || { echo '当前终端找不到 conda；请使用 AutoDL 的 conda 终端。'; exit 1; }
if ! conda run -n cv python -c 'import sys; assert sys.version_info[:2] == (3, 12)' >/dev/null 2>&1; then
  if conda env list --json | python -c 'import json,sys; from pathlib import Path; sys.exit(0 if any(Path(p).name == "cv" for p in json.load(sys.stdin)["envs"]) else 1)'; then
    echo '已存在 cv，但不是 Python 3.12。请保留原环境，先为本工程准备 Python 3.12 的 cv 环境。'
    exit 1
  fi
  conda env create -f environment.yml
fi
if ! conda run -n cv python -c 'import torch,torchvision; assert torch.__version__ == "2.7.1+cu128" and torchvision.__version__ == "0.22.1+cu128"' >/dev/null 2>&1; then
  conda run --no-capture-output -n cv python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
fi
conda run --no-capture-output -n cv python -m pip install --no-cache-dir -r requirements.txt
conda run --no-capture-output -n cv python -m pip check
conda run --no-capture-output -n cv python -B scripts/fetch_pretrained.py
conda run --no-capture-output -n cv python -m ipykernel install --user --name cv --display-name 'Python (conda cv)'
conda run --no-capture-output -n cv python -c 'import torch; assert torch.cuda.is_available(), "CUDA 不可用，请检查 AutoDL GPU、驱动和当前环境"; print("PyTorch:",torch.__version__,"CUDA:",torch.version.cuda,"GPU:",torch.cuda.get_device_name(0)); print("环境已准备；在 VS Code 打开 BUSI_UGBSegNet.ipynb，选择 Python (conda cv)。训练由你手动运行。")'
