import sqlalchemy
import pandas as pd
import numpy as np
from pandas.io.common import file_exists
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base

from analyzer import Analyzer
from database import DatabaseManager
from processor import TestProcessor
from dataloader import TrainingLoader, IdealFunctionLoader, DataLoader
from visualizer import Visualizer

if __name__ == "__main__":
    print("Main starting...")

    print("sqlalchemy: {}".format(sqlalchemy.__version__))

    try:

        TRAIN_FILE_PATH = 'data/train.csv'
        IDEAL_FILE_PATH = 'data/ideal.csv'
        TEST_FILE_PATH = 'data/test.csv'

        OUTPUT_FILE_PATH = 'output/test_results.csv'

        db_manager = DatabaseManager(db_name="idealfunction.db")

        #Loading data from CSV files into DataFrames
        train_loader = TrainingLoader(TRAIN_FILE_PATH)
        ideal_loader = IdealFunctionLoader(IDEAL_FILE_PATH)
        test_loader = DataLoader(TEST_FILE_PATH)

        training_df = train_loader.load_data()
        ideal_df = ideal_loader.load_data()

        db_manager.write_data_with_x_index(training_df, 'training_data', if_exists='replace')
        db_manager.write_data_with_x_index(ideal_df, 'ideal_functions', if_exists='replace')

        print("\n--- Training Data Head (from DB) ---")
        print(db_manager.read_table_to_dataframe("training_data").head())

        print("\n--- Ideal Functions Data Head (from DB) ---")
        print(db_manager.read_table_to_dataframe("ideal_functions").head())

        print("\nPhase 1 (Data Loading) is complete.")

        analyzer = Analyzer(db_manager)
        best_fit_ranking, max_deviations = analyzer.run_analysis()

        print("\n---- Analysis results----")
        print(f"Best-fit function ranking: {best_fit_ranking}")
        print(f"Max deviations: {max_deviations}")

        print("\nPhase 2 (Data Analysis) is complete.")
        print("\nPhase 3: Test Data Processing...")

        processor = TestProcessor(
            db_manager=db_manager,
            ideal_df=ideal_df,
            max_deviations=max_deviations
        )

        processor.process_test_file(TEST_FILE_PATH, OUTPUT_FILE_PATH)

        print("\n--- Test Results (from DB) ---")
        print(db_manager.read_table_to_dataframe("test_results").head())

        print("\nPhase 3 is complete.")

        visualizer = Visualizer(db_manager, best_fit_ranking)
        visualizer.generate_and_save_plots(
            test_file_path=TEST_FILE_PATH,
        )

        print("\nPhase 4 (Visualization) is complete.")
        print(f"\nAll tasks complete.")

    except FileNotFoundError as e:
        print(f"Error: File not found at {e.filename}")
    except Exception as e:
        print(f"Error: {e}")













