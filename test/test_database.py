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
