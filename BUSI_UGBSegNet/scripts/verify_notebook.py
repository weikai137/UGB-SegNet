"""Execute only tagged preparation/display cells in an isolated conda cv kernel."""
import ast
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["UGB_VERIFY_ONLY"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "BUSI_UGBSegNet.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == "code":
        ast.parse(cell.source)
selected = [cell for cell in notebook.cells if cell.cell_type == "code" and not cell.metadata.get("manual_compute", True)]
assert len(selected) == 7
qa = nbformat.v4.new_notebook(cells=selected)
client = NotebookClient(qa, kernel_name="cv", timeout=120, resources={"metadata": {"path": str(ROOT)}})
client.execute()
executed = {cell.id: cell for cell in qa.cells}
for cell in notebook.cells:
    if cell.id in executed:
        cell.outputs = executed[cell.id].outputs
        cell.execution_count = executed[cell.id].execution_count
        assert all(output.output_type != "error" for output in cell.outputs)
notebook.metadata["verification"] = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "host_python": sys.executable, "kernel": "cv", "safe_cells_executed": len(selected),
    "manual_compute_cells_deferred": 6, "gpu_training_started": False,
    "note": "Only environment, parameters, data audit, training samples, function definitions and existing-results views were executed. No model training/test inference."}
nbformat.validate(notebook)
nbformat.write(notebook, path)
print("NOTEBOOK PASS: 7 preparation/display cells executed; 6 manual compute cells left unexecuted.", flush=True)
