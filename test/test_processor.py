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


def test_processor_boundary_equality_is_mapped():
    ideal_df = pd.DataFrame({'X': [0.0], 'y1': [0.0]})
    processor = DataMapper(
        db_manager=MagicMock(),
        ideal_df=ideal_df,
        max_deviations={'y1': 0.5},
        best_fit_ranking={'y1_train': [('y1', 0.1)]},
    )
    threshold = 0.5 * np.sqrt(2)

    processor.map_point(x_test=0.0, y_test=threshold)

    assert len(processor.results) == 1
    assert processor.results[0]['No_Ideal_Func'] == 'y1'
    assert processor.results[0]['Delta_Y'] == pytest.approx(threshold)


def test_processor_multimatch_chooses_lowest_deviation():
    ideal_df = pd.DataFrame({'X': [0.0], 'y1': [0.0], 'y2': [0.2]})
    processor = DataMapper(
        db_manager=MagicMock(),
        ideal_df=ideal_df,
        max_deviations={'y1': 0.5, 'y2': 0.5},
        best_fit_ranking={'y1_train': [('y1', 0.1)], 'y2_train': [('y2', 0.1)]},
    )

    processor.map_point(x_test=0.0, y_test=0.05)   # dev y1=0.05, y2=0.15 -> y1
    assert processor.results[0]['No_Ideal_Func'] == 'y1'
    assert processor.results[0]['Delta_Y'] == pytest.approx(0.05)

    processor.map_point(x_test=0.0, y_test=0.19)   # dev y1=0.19, y2=0.01 -> y2
    assert processor.results[1]['No_Ideal_Func'] == 'y2'
    assert processor.results[1]['Delta_Y'] == pytest.approx(0.01)


def test_processor_unknown_x_is_unmapped_but_counted():
    ideal_df = pd.DataFrame({'X': [0.0], 'y1': [0.0]})
    processor = DataMapper(
        db_manager=MagicMock(),
        ideal_df=ideal_df,
        max_deviations={'y1': 0.5},
        best_fit_ranking={'y1_train': [('y1', 0.1)]},
    )

    processor.map_point(x_test=999.0, y_test=0.0)

    assert processor.results == []
    assert processor.total_processed == 1


def _stats_processor():
    ideal_df = pd.DataFrame({'X': [0.0, 10.0], 'y1': [0.0, 10.0], 'y2': [1.0, 11.0]})
    return DataMapper(
        db_manager=MagicMock(),
        ideal_df=ideal_df,
        max_deviations={'y1': 0.3, 'y2': 0.3},
        best_fit_ranking={'y1_train': [('y1', 0.1)], 'y2_train': [('y2', 0.1)]},
    )


def test_compute_mapping_statistics_counts_and_csv(tmp_path):
    processor = _stats_processor()
    processor.map_point(0.0, 0.1)     # mapped -> y1
    processor.map_point(10.0, 11.1)   # mapped -> y2
    processor.map_point(0.0, 5.0)     # unmapped (too far from both)
    processor.map_point(99.0, 0.0)    # unmapped (unknown x)

    out = tmp_path / "mapping_statistics.csv"
    stats = processor.compute_mapping_statistics(output_path=str(out))

    assert stats['total_processed'] == 4
    assert stats['mapped'] == 2
    assert stats['unmapped'] == 2
    assert stats['rejection_rate'] == pytest.approx(0.5)

    df = pd.read_csv(out)
    assert list(df.columns) == [
        'Ideal_Function', 'Original_Train_Func', 'Mapped_Count',
        'Total_Processed', 'Unmapped_Count', 'Rejection_Rate',
        'Min_Delta_Y', 'Max_Delta_Y', 'Mean_Delta_Y', 'Std_Delta_Y',
    ]
    overall = df[df['Ideal_Function'] == 'OVERALL'].iloc[0]
    assert overall['Total_Processed'] == pytest.approx(4)
    assert overall['Mapped_Count'] == pytest.approx(2)
    assert overall['Unmapped_Count'] == pytest.approx(2)
    assert overall['Rejection_Rate'] == pytest.approx(0.5)


def test_compute_mapping_statistics_empty_results(tmp_path):
    processor = _stats_processor()
    out = tmp_path / "empty_mapping_statistics.csv"

    stats = processor.compute_mapping_statistics(output_path=str(out))

    assert stats['total_processed'] == 0
    assert stats['mapped'] == 0
    assert stats['unmapped'] == 0
    assert stats['rejection_rate'] == 0.0

    df = pd.read_csv(out)
    assert list(df.columns) == [
        'Ideal_Function', 'Original_Train_Func', 'Mapped_Count',
        'Total_Processed', 'Unmapped_Count', 'Rejection_Rate',
        'Min_Delta_Y', 'Max_Delta_Y', 'Mean_Delta_Y', 'Std_Delta_Y',
    ]
    assert len(df) == 1
    assert df.iloc[0]['Ideal_Function'] == 'OVERALL'