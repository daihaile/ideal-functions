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
    ideal_data = {
        'X': [-10.0, 0.0, 10.0],
        'y1': [-10.0, 0.0, 10.0],
    }
    ideal_df = pd.DataFrame(ideal_data)

    max_deviations = {'y1': 0.5}

    best_fit_ranking = {'y1_train': [('y1', 0.1)]}  # {'train_col': [('ideal_col', ssq)]}

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

    '''
            Max Deviation = 0.5
            Sqrt(2) = ~1.414
            Mapping Threshold = 0.5 * 1.414 = 0.707
            (x=0.0, y=0.6). Ideal value is 0.0.
            Deviation = abs(0.6 - 0.0) = 0.6
            0.6 < 0.707 -> should be mapped.
    '''


    processor.map_point(x_test=0.0, y_test=0.6)

    assert len(processor.results) == 1
    assert processor.results[0]['X'] == 0.0
    assert processor.results[0]['Y'] == 0.6
    assert processor.results[0]['No_Ideal_Func'] == 'y1'
    assert processor.results[0]['Delta_Y'] == pytest.approx(0.6)

    # deviation = 0.8 > 0.707 -> not mapped
    processor.map_point(x_test=10.0, y_test=10.8)

    assert len(processor.results) == 1