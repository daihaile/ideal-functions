import pandas as pd
from sqlalchemy import create_engine, Column, Float, String, Integer
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Column name constants for TestResults — single source of truth
COL_X_TEST = 'X (test func)'
COL_Y_TEST = 'Y (test func)'
COL_DELTA_Y_TEST = 'Delta Y (test func)'
COL_NO_IDEAL_FUNC = 'No. of ideal func'
COL_Y_IDEAL = 'Y_Ideal'
COL_MAPPING_THRESHOLD = 'Mapping_Threshold'
COL_ORIGINAL_TRAIN_FUNC = 'Original_Train_Func'

class TrainingData(Base):
    """
    SQLAlchemy model for the training data table
    """
    __tablename__ = "training_data"
    X = Column(Float, primary_key=True)
    Y1 = Column(Float)
    Y2 = Column(Float)
    Y3 = Column(Float)
    Y4 = Column(Float)

class IdealFunctions(Base):
    """
    SQLAlchemy model for the ideal functions table
    """
    __tablename__ = 'ideal_functions'

    X = Column(Float, primary_key=True)

    for i in range(1, 51):
        vars()[f'Y{i}'] = Column(Float)


class TestResults(Base):
    """
    SQLAlchemy model for the test results table
    """
    __tablename__ = 'test_results'
    id = Column(Integer, primary_key=True, autoincrement=True)

    X = Column(COL_X_TEST, Float)
    Y = Column(COL_Y_TEST, Float)
    Delta_Y = Column(COL_DELTA_Y_TEST, Float)
    No_Ideal_Func = Column(COL_NO_IDEAL_FUNC, String)

    Y_Ideal = Column(COL_Y_IDEAL, Float)
    Mapping_Threshold = Column(COL_MAPPING_THRESHOLD, Float)
    Original_Train_Func = Column(COL_ORIGINAL_TRAIN_FUNC, String)

class DatabaseManager:
    """
    helper class for managing database operations
    """

    def __init__(self, db_name="idealfunction.db"):
        """Init database engine."""
        self.db_path = db_name
        self.engine = create_engine(f"sqlite:///{db_name}")
        Base.metadata.create_all(self.engine)
        print(f"Database '{self.db_path}' and tables created.")

    def write_data_with_x_index(self, df, table_name, if_exists='replace'):
        """Writes a DataFrame using 'X' as the primary key """
        try:
            with self.engine.connect() as conn:
                df_with_index = df.set_index('X')
                df_with_index.to_sql(table_name, conn, if_exists=if_exists, index=True, index_label='X')
            print(f"Successfully wrote data to '{table_name}'.")
        except Exception as e:
            print(f"Error writing to database: {e}")
            raise

    def write_test_results(self, results_df, if_exists='replace'):
        """Writes the test results DataFrame to the 'test_results' table"""
        try:
            with self.engine.connect() as conn:
                results_df.to_sql('test_results', conn, if_exists=if_exists, index=False)
            print(f"Successfully wrote data to 'test_results'.")
        except Exception as e:
            print(f"Error writing test results to database: {e}")
            raise

    def read_table_to_dataframe(self, table_name):
        """Reads a table into a panda sDataFrame."""
        print(f"Reading data from {table_name}...")
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_table(table_name, conn)
            print(f"Data read from {table_name}.")
            return df
        except Exception as e:
            print(f"Error reading data from {table_name}: {e}")
            return pd.DataFrame()
