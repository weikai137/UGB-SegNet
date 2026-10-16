"""Clear notebook outputs before Git publication; preserve code and disk training results."""
import json
from pathlib import Path


def clear_outputs(path):
    path = Path(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    notebook["metadata"].pop("verification", None)
    notebook["metadata"].pop("widgets", None)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    print("Notebook 输出已清空；代码及磁盘上的训练权重、日志、报告保留。")


if __name__ == "__main__":
    clear_outputs(Path(__file__).resolve().parents[1] / "BUSI_UGBSegNet.ipynb")
