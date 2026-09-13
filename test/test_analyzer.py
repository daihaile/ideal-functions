import pandas as pd
import pytest
from unittest.mock import MagicMock
import numpy as np
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


def test_analyzer_export_summary_table(mock_db_manager, tmp_path):
    analyzer = Analyzer(mock_db_manager)
    analyzer.run_analysis()

    out = tmp_path / "analysis_summary.csv"
    summary = analyzer.export_summary_table(output_path=str(out))

    assert list(summary.columns) == [
        'Training_Col', 'Best_Ideal', 'SSQ_Best', 'Second_Best', 'SSQ_Second',
        'Third_Best', 'SSQ_Third', 'Max_Deviation', 'Threshold_sqrt2',
    ]
    row = summary[summary['Training_Col'] == 'y1'].iloc[0]
    assert row['Best_Ideal'] == 'y2'
    assert row['SSQ_Best'] == pytest.approx(0.0)
    assert row['Second_Best'] == 'y3'
    assert row['SSQ_Second'] == pytest.approx(0.03)
    assert row['Third_Best'] == 'y1'
    assert row['SSQ_Third'] == pytest.approx(0.83)
    assert row['Max_Deviation'] == pytest.approx(0.0)
    assert row['Threshold_sqrt2'] == pytest.approx(0.0)
    assert out.exists()


def test_analyzer_export_summary_table_threshold_uses_sqrt2(tmp_path):
    """Threshold_sqrt2 must equal Max_Deviation * sqrt(2) for a non-zero deviation."""
    train_df = pd.DataFrame({'X': [1, 2, 3], 'y1': [1.1, 2.1, 3.1]})
    ideal_df = pd.DataFrame({'X': [1, 2, 3], 'y2': [1.0, 2.0, 3.0], 'y3': [1.0, 2.0, 4.0]})
    mock_db = MagicMock()
    mock_db.read_table_to_dataframe.side_effect = [train_df, ideal_df]

    analyzer = Analyzer(mock_db)
    analyzer.run_analysis()

    out = tmp_path / "analysis_summary_sqrt2.csv"
    summary = analyzer.export_summary_table(output_path=str(out))

    row = summary.iloc[0]
    assert row['Best_Ideal'] == 'y2'
    assert row['Max_Deviation'] == pytest.approx(0.1)          # non-zero, unlike the shared fixture
    assert row['Threshold_sqrt2'] == pytest.approx(row['Max_Deviation'] * np.sqrt(2))
    assert out.exists()