import os
import sys

import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from database import DatabaseManager


def _training_df():
    return pd.DataFrame({
        'X': [1.0, 2.0, 3.0],
        'y1': [1.0, 2.0, 3.0],
        'y2': [2.0, 3.0, 4.0],
        'y3': [3.0, 4.0, 5.0],
        'y4': [4.0, 5.0, 6.0],
    })


def _ideal_df():
    return pd.DataFrame({
        'X': [1.0, 2.0, 3.0],
        'y1': [1.1, 2.1, 3.1],
        'y2': [1.2, 2.2, 3.2],
        'y3': [1.3, 2.3, 3.3],
    })


def _pk_of(info, column_name):
    return [row for row in info if row[1] == column_name][0][5]


def test_training_data_x_is_primary_key(tmp_path):
    db = DatabaseManager(db_name=str(tmp_path / "t.db"))
    db.write_data_with_x_index(_training_df(), 'training_data')

    with db.engine.connect() as conn:
        info = conn.exec_driver_sql("PRAGMA table_info(training_data)").fetchall()

    assert len(info) == 5
    assert _pk_of(info, 'X') == 1


def test_ideal_functions_x_is_primary_key(tmp_path):
    db = DatabaseManager(db_name=str(tmp_path / "t.db"))
    db.write_data_with_x_index(_ideal_df(), 'ideal_functions')

    with db.engine.connect() as conn:
        info = conn.exec_driver_sql("PRAGMA table_info(ideal_functions)").fetchall()

    assert _pk_of(info, 'X') == 1


def test_duplicate_x_raises_integrity_error(tmp_path):
    db = DatabaseManager(db_name=str(tmp_path / "t.db"))
    db.write_data_with_x_index(_training_df(), 'training_data')

    with pytest.raises(IntegrityError):
        db.write_data_with_x_index(_training_df(), 'training_data', if_exists='append')


def test_read_training_data_keeps_lowercase_columns(tmp_path):
    db = DatabaseManager(db_name=str(tmp_path / "t.db"))
    db.write_data_with_x_index(_training_df(), 'training_data')

    df = db.read_table_to_dataframe('training_data')

    assert list(df.columns) == ['X', 'y1', 'y2', 'y3', 'y4']


def test_mapping_outputs_share_rows_and_use_distinct_schemas(tmp_path):
    from processor import DataMapper

    db = DatabaseManager(db_name=str(tmp_path / "results.db"))
    test_path = tmp_path / "test.csv"
    output_path = tmp_path / "mapped.csv"
    sql_columns = ["X (test func)", "Y (test func)", "Delta Y (test func)", "No. of ideal func"]
    csv_columns = ["X", "Y", "Delta_Y", "No_Ideal_Func", "Y_Ideal", "Mapping_Threshold", "Original_Train_Func"]
    ideal = pd.DataFrame({"X": [1.0], "y1": [2.0]})

    for y, expected_count in [(2.25, 2), (20.0, 0)]:
        pd.DataFrame({"x": [1.0, 1.0], "y": [y, y]}).to_csv(test_path, index=False)
        mapper = DataMapper(db, ideal, {"y1": 0.5}, {"y_train": [("y1", 0.0)]})
        details = mapper.process_test_file(test_path, output_path)
        stored = db.read_table_to_dataframe("test_results")
        exported = pd.read_csv(output_path)
        assert list(stored.columns) == sql_columns
        assert list(exported.columns) == csv_columns
        assert len(stored) == len(exported) == len(details) == expected_count
        if expected_count:
            assert list(stored.itertuples(index=False, name=None)) == [(1.0, 2.25, 0.25, "y1")] * 2
            assert details["Mapping_Threshold"].iloc[0] == pytest.approx(0.5 * 2**0.5)
