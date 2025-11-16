import pandas as pd
import pytest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from processor import DataMapper


@pytest.fixture
def processor_inputs():
    """
    This fixture creates all the 'GIVEN' data needed to initialize
    a TestProcessor.
    """
    # 1. GIVEN: Fake ideal function data
    ideal_data = {
        'X': [-10.0, 0.0, 10.0],
        'y1': [-10.0, 0.0, 10.0],  # Our "winner" function
    }
    ideal_df = pd.DataFrame(ideal_data)

    # 2. GIVEN: Fake analysis results (from Analyzer)
    # The max deviation for our 'y1' (train) vs 'y1' (ideal) was 0.5
    max_deviations = {'y1': 0.5}

    # The 'y1' (train) matched with 'y1' (ideal)
    best_fit_ranking = {'y1_train': [('y1', 0.1)]}  # {'train_col': [('ideal_col', ssq)]}

    # 3. GIVEN: A fake database manager
    mock_db = MagicMock()

    return {
        "db_manager": mock_db,
        "ideal_df": ideal_df,
        "max_deviations": max_deviations,
        "best_fit_ranking": best_fit_ranking
    }


def test_processor_map_point(processor_inputs):
    """
    Tests the core mapping logic of the TestProcessor.
    """
    processor = DataMapper(
        db_manager=processor_inputs["db_manager"],
        ideal_df=processor_inputs["ideal_df"],
        max_deviations=processor_inputs["max_deviations"],
        best_fit_ranking=processor_inputs["best_fit_ranking"]
    )

    # --- Our Rule ---
    # Max Deviation = 0.5
    # Sqrt(2) = ~1.414
    # Mapping Threshold = 0.5 * 1.414 = 0.707

    # 2. WHEN: We test a point that should PASS
    # (x=0.0, y=0.6). Ideal value is 0.0.
    # Deviation = abs(0.6 - 0.0) = 0.6
    # 0.6 is LESS than 0.707, so it should be mapped.
    processor.map_point(x_test=0.0, y_test=0.6)

    # 3. THEN: Check that the result was saved
    assert len(processor.results) == 1
    assert processor.results[0]['X'] == 0.0
    assert processor.results[0]['Y'] == 0.6
    assert processor.results[0]['No_Ideal_Func'] == 'y1'
    assert processor.results[0]['Delta_Y'] == pytest.approx(0.6)

    # 4. WHEN: We test a point that should FAIL
    # (x=10.0, y=10.8). Ideal value is 10.0.
    # Deviation = abs(10.8 - 10.0) = 0.8
    # 0.8 is GREATER than 0.707, so it should be rejected.
    processor.map_point(x_test=10.0, y_test=10.8)

    # 5. THEN: Check that no new result was added
    assert len(processor.results) == 1  # Still 1, no new result