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

    # 3. Create a "mock" db_manager object
    mock_db = MagicMock()

    # 4. Tell the mock object how to behave.
    # When `read_table_to_dataframe` is called with "training_data",
    # return our fake train_df.
    mock_db.read_table_to_dataframe.side_effect = lambda table_name: \
        train_df if table_name == "training_data" else \
            ideal_df if table_name == "ideal_functions" else \
                pd.DataFrame()  # Return empty for any other call

    return mock_db


def test_analyzer_run_analysis(mock_db_manager):
    """
    This is our unit test. It uses the fake db_manager
    to test the Analyzer's logic.
    """

    # 1. GIVEN: Create an Analyzer instance using our fake db_manager
    # The Analyzer *thinks* it's talking to a real database.
    analyzer = Analyzer(mock_db_manager)

    # 2. WHEN: We run the analysis
    best_fit_ranking, max_deviations = analyzer.run_analysis()

    # 3. THEN: We check (assert) that the results are correct.

    # Check that 'y1' was found and analyzed
    assert 'y1' in best_fit_ranking

    # Check that the #1 best fit is 'y2' (our perfect match)
    assert best_fit_ranking['y1'][0][0] == 'y2'

    # Check that the Sum of Squares for 'y2' is 0
    assert best_fit_ranking['y1'][0][1] == pytest.approx(0.0)

    # Check that the #2 best fit is 'y3' (our good match)
    assert best_fit_ranking['y1'][1][0] == 'y3'
    assert best_fit_ranking['y1'][1][1] == pytest.approx(0.03)

    # Check that the #3 best fit is 'y1' (our bad match)
    assert best_fit_ranking['y1'][2][0] == 'y1'
    assert best_fit_ranking['y1'][2][1] == pytest.approx(0.83)

    # Check that the max deviation for the best fit ('y2') is 0
    assert 'y2' in max_deviations
    assert max_deviations['y2'] == pytest.approx(0.0)