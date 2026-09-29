"""
Notebook validation script: Runs all code cells of 01_data_understanding_eda.ipynb and 02_feature_engineering_ml.ipynb
and verifies zero execution errors.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


def run_notebook(nb_path: Path):
    print(f"\n==========================================")
    print(f" EXECUTING NOTEBOOK: {nb_path.name}")
    print(f"==========================================")
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Global execution namespace
    exec_namespace = {"__file__": str(nb_path), "__name__": "__main__"}

    code_cells = [cell for cell in nb_data.get("cells", []) if cell.get("cell_type") == "code"]
    print(f"Found {len(code_cells)} code cells to execute.")

    for i, cell in enumerate(code_cells, 1):
        source = "".join(cell.get("source", []))
        # Filter out display() or IPython specific calls if not in IPython environment
        lines = []
        for line in source.splitlines():
            if line.strip().startswith("%") or line.strip().startswith("!"):
                continue
            lines.append(line)
        clean_code = "\n".join(lines)

        try:
            exec(clean_code, exec_namespace)
            print(f"  [CELL {i}/{len(code_cells)}] OK")
        except Exception as e:
            print(f"  [CELL {i}/{len(code_cells)}] ERROR: {e}")
            raise e

    print(f"[OK] Notebook '{nb_path.name}' executed with 0 errors!")


if __name__ == "__main__":
    run_notebook(PROJECT_ROOT / "notebooks" / "01_data_understanding_eda.ipynb")
    run_notebook(PROJECT_ROOT / "notebooks" / "02_feature_engineering_ml.ipynb")
    print("\n==========================================")
    print(" ALL JUPYTER NOTEBOOKS PASSED WITH 0 ERRORS!")
    print("==========================================")
