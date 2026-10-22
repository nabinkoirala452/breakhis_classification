"""
Turn a "# %% [Cell N]" script into a Kaggle notebook + kernel-metadata.json.

usage: python build_notebook.py <script.py> <out_dir> <owner/slug> <dataset> [<dataset> ...]
then:  kaggle kernels push -p <out_dir> --accelerator NvidiaTeslaT4
"""
import json
import re
import sys
from pathlib import Path

script, out_dir, kernel_id, datasets = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
slug = kernel_id.split("/")[1]
src = Path(script).read_text(encoding="utf-8")
parts = re.split(r"(?m)^(?=# %% \[Cell )", src)


def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [{"cell_type": "markdown", "metadata": {},
          "source": [f"# {slug}\n", f"Generated from `{Path(script).name}`.\n"]}]
cells += [code_cell(p) for p in parts if p.strip()]
compile("\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code"), slug, "exec")

out_dir.mkdir(parents=True, exist_ok=True)
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
(out_dir / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
meta = {"id": kernel_id, "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "keywords": ["gpu"], "dataset_sources": datasets,
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}
(out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"{len(cells)} cells -> {out_dir} ({kernel_id})")
