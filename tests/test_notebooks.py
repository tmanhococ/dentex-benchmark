import json
from pathlib import Path


def test_notebooks_exist_and_valid():
    nb_dir = Path("notebooks")
    assert nb_dir.exists(), "notebooks/ directory does not exist"

    expected_notebooks = [
        "01_colab_run_all.ipynb",
        "02_kaggle_run_all.ipynb",
        "03_local_benchmark.ipynb",
        "dentex_eda_analysis.ipynb",
        "dentex_inference_error_analysis.ipynb",
    ]

    for name in expected_notebooks:
        nb_path = nb_dir / name
        assert nb_path.exists(), f"Expected notebook {name} does not exist"

        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "cells" in data, f"{name} missing 'cells' key"
        assert "metadata" in data, f"{name} missing 'metadata' key"
        assert data.get("nbformat") == 4, f"{name} is not nbformat 4"
        assert len(data["cells"]) > 0, f"{name} has no cells"

        # Verify cell schema
        for i, cell in enumerate(data["cells"]):
            assert "cell_type" in cell, f"Cell {i} in {name} missing cell_type"
            assert "source" in cell, f"Cell {i} in {name} missing source"


def test_notebooks_content_requirements():
    nb_dir = Path("notebooks")
    assert nb_dir.exists()

    # 01 Colab checks
    colab_path = nb_dir / "01_colab_run_all.ipynb"
    assert colab_path.exists()
    colab_text = colab_path.read_text(encoding="utf-8")
    assert "drive" in colab_text or "google.colab" in colab_text
    assert "setup_env" in colab_text or "dentex_bench" in colab_text
    assert "download_and_prepare" in colab_text
    assert "train" in colab_text
    assert "evaluate" in colab_text
    assert "git clone" in colab_text

    # 02 Kaggle checks
    kaggle_path = nb_dir / "02_kaggle_run_all.ipynb"
    assert kaggle_path.exists()
    kaggle_text = kaggle_path.read_text(encoding="utf-8")
    assert "kaggle" in kaggle_text.lower()
    assert "dentex_bench" in kaggle_text
    assert "train" in kaggle_text
    assert "evaluate" in kaggle_text
    assert "git clone" in kaggle_text

    # 03 Local checks
    local_path = nb_dir / "03_local_benchmark.ipynb"
    assert local_path.exists()
    local_text = local_path.read_text(encoding="utf-8")
    assert "dentex_bench" in local_text
    assert "project_root.name == 'notebooks'" in local_text or 'project_root.name == "notebooks"' in local_text
    assert "build_model()" in local_text
    assert "runner.predict([img_tensor])" in local_text
    assert "visualiz" in local_text.lower() or "matplotlib" in local_text.lower()
    for cat in ["Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth"]:
        assert cat in local_text


def test_documentation_and_requirements():
    req_path = Path("requirements.txt")
    assert req_path.exists(), "requirements.txt does not exist"
    req_text = req_path.read_text(encoding="utf-8")
    for pkg in ["torch", "torchvision", "transformers", "pycocotools", "albumentations"]:
        assert pkg in req_text

    readme_path = Path("README.md")
    assert readme_path.exists(), "README.md does not exist"
    readme_text = readme_path.read_text(encoding="utf-8")
    assert "DENTEX" in readme_text
    for cat in ["Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth"]:
        assert cat in readme_text
    for model in ["Faster R-CNN", "DINO", "HierarchicalDet"]:
        assert model in readme_text


def test_dentex_eda_analysis_notebook():
    nb_path = Path("notebooks/dentex_eda_analysis.ipynb")
    assert nb_path.exists()
    text = nb_path.read_text(encoding="utf-8")
    for key in [
        "instances_train.json",
        "instances_val.json",
        "Spatial Heatmap",
        "Co-occurrence",
        "COCO",
        "Caries",
        "Deep Caries",
        "Periapical Lesion",
        "Impacted Tooth",
    ]:
        assert key in text, f"Missing {key} in dentex_eda_analysis.ipynb"


def test_dentex_inference_error_analysis_notebook():
    nb_path = Path("notebooks/dentex_inference_error_analysis.ipynb")
    assert nb_path.exists()
    text = nb_path.read_text(encoding="utf-8")
    for key in [
        "HierarchicalNet",
        "DiffusionNet",
        "Confusion Matrix",
        "Success Profiling",
        "Top 5 Failure Modes",
        "Ground Truth",
        "best.pth",
    ]:
        assert key in text, f"Missing {key} in dentex_inference_error_analysis.ipynb"


def test_notebook_code_cells_syntax():
    nb_dir = Path("notebooks")
    for nb in nb_dir.glob("*.ipynb"):
        with open(nb, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                # Strip out IPython shell magics (lines starting with ! or %)
                clean_lines = []
                for line in source.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("!") or stripped.startswith("%"):
                        clean_lines.append("# " + line)
                    else:
                        clean_lines.append(line)
                clean_code = "\n".join(clean_lines)
                try:
                    compile(clean_code, f"{nb.name}:cell_{i}", "exec")
                except SyntaxError as e:
                    raise AssertionError(f"Syntax error in {nb.name} cell {i}: {e}")

