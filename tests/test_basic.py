"""Basic tests for BlueBike project."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_requirements_exists():
    assert os.path.exists("requirements.txt")


def test_src_structure():
    assert os.path.isdir("src")
    assert os.path.isdir("src/models")
    assert os.path.isdir("src/preprocessing")
    assert os.path.isdir("src/visualization")


def test_results_directory():
    assert os.path.isdir("results")


def test_model_files_exist():
    assert os.path.exists("src/models/baseline.py")
    assert os.path.exists("src/models/xgboost_model.py")
    assert os.path.exists("src/models/xgboost_poi_only.py")


def test_import_modules():
    import pandas
    import numpy
    import xgboost


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
