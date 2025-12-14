import pandas as pd
import pytest
from unittest.mock import MagicMock
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from analyzer import Analyzer

@pytest.fixture
def mock_db_manager():
    """Create a fake DatabaseManager for Analyzer unit tests.

    returns small in-memory DataFrames for:
      - "training_data"
      - "ideal_functions"
    """

    train_data = {
        'X': [1.0, 2.0, 3.0],
        'y1': [1.1, 2.1, 3.1]
    }
    train_df = pd.DataFrame(train_data)

    ideal_data = {
        'X': [1.0, 2.0, 3.0],
        'y1': [1.0, 2.0, 4.0],  # Bad fit (SSQ = (0.1)^2 + (0.1)^2 + (-0.9)^2 = 0.83)
        'y2': [1.1, 2.1, 3.1],  # Perfect fit (SSQ = 0)
        'y3': [1.0, 2.0, 3.0]  # Good fit (SSQ = (0.1)^2 + (0.1)^2 + (0.1)^2 = 0.03)
    }
    ideal_df = pd.DataFrame(ideal_data)

    mock_db = MagicMock()


    mock_db.read_table_to_dataframe.side_effect = lambda table_name: \
        train_df if table_name == "training_data" else \
            ideal_df if table_name == "ideal_functions" else \
                pd.DataFrame()

    return mock_db


def test_analyzer_run_analysis(mock_db_manager):
    """
    unit test using the fake db_manager to test the Analyzer's logic.
    """


    analyzer = Analyzer(mock_db_manager)

    best_fit_ranking, max_deviations = analyzer.run_analysis()

    assert 'y1' in best_fit_ranking
    assert best_fit_ranking['y1'][0][0] == 'y2'
    assert best_fit_ranking['y1'][0][1] == pytest.approx(0.0)
    assert best_fit_ranking['y1'][1][0] == 'y3'
    assert best_fit_ranking['y1'][1][1] == pytest.approx(0.03)
    assert best_fit_ranking['y1'][2][0] == 'y1'
    assert best_fit_ranking['y1'][2][1] == pytest.approx(0.83)
    assert 'y2' in max_deviations
    assert max_deviations['y2'] == pytest.approx(0.0)