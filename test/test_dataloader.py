import os
import sys

import pandas as pd
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from dataloader import DataLoader
from exceptions.DataMismatcherror import DataMismatchError


def test_dataloader_raises_datamismatch(tmp_path):
    # Create a CSV without 'X' or 'x' to hit exception
    p = tmp_path / "bad.csv"
    pd.DataFrame({"not_x": [1, 2, 3], "y": [10, 20, 30]}).to_csv(p, index=False)

    loader = DataLoader(str(p))

    with pytest.raises(DataMismatchError):
        loader.load_data()


def test_dataloader_renames_lowercase_x_to_X(tmp_path):
    p = tmp_path / "ok.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}).to_csv(p, index=False)

    loader = DataLoader(str(p))
    df = loader.load_data()

    assert "X" in df.columns
    assert "x" not in df.columns