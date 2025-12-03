"""
Basic tests for BlueBike project.
These tests verify that core components work correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_requirements_exists():
    """Test that requirements.txt exists."""
    assert os.path.exists("requirements.txt")


def test_src_structure():
    """Test that source code structure is correct."""
    assert os.path.isdir("src")
    assert os.path.isdir("src/models")
    assert os.path.isdir("src/preprocessing")
    assert os.path.isdir("src/visualization")


def test_results_directory():
    """Test that results directory exists."""
    assert os.path.isdir("results")


def test_model_files_exist():
    """Test that model source files exist."""
    assert os.path.exists("src/models/baseline.py")
    assert os.path.exists("src/models/xgboost_model.py")
    assert os.path.exists("src/models/xgboost_poi_only.py")


def test_import_modules():
    """Test that key modules can be imported."""
    import pandas
    import numpy
    import xgboost
    assert True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
