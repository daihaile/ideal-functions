import pandas as pd
import numpy as np
from database import DatabaseManager


class DataMapper:
    """
    Handles loading, processing, and mapping the test data.
    """

    def __init__(self, db_manager: DatabaseManager, ideal_df: pd.DataFrame, max_deviations: dict, best_fit_ranking: dict):
        """
        Initializes the processor

        """
        self.db_manager = db_manager
        self.chosen_ideal_cols = list(max_deviations.keys())
        self.ideal_data = ideal_df[['X'] + self.chosen_ideal_cols].set_index('X')
        self.max_deviations = max_deviations
        self.sqrt_2_factor = np.sqrt(2)
        self.results = []

        self.ideal_to_train_map = {}
        for train_col, top_fits in best_fit_ranking.items():
            ideal_col = top_fits[0][0]
            self.ideal_to_train_map[ideal_col] = train_col

    def process_test_file(self, test_file_path: str, output_csv_path: str = None):
        """
        Reads the test CSV file line-by-line, processes each point and saves the results to the database.
        """
        print(f"Starting to process test file: {test_file_path}...")

        try:

            for row in pd.read_csv(test_file_path, chunksize=1):
                row = row.rename(columns={'x': 'X', 'y': 'Y'})

                x = row['X'].values[0]
                y = row['Y'].values[0]
                self.map_point(x, y)

            print(f"Finished processing test file. Found {len(self.results)} mappable points.")

            if self.results:
                results_df = pd.DataFrame(self.results)

                if output_csv_path:
                    try:
                        results_df.to_csv(output_csv_path, index=False)
                        print(f"Successfully saved mapped results to {output_csv_path}")
                    except Exception as e:
                        print(f"Error saving to CSV: {e}")

                results_df = results_df.rename(columns={
                    'X': 'X (test func)',
                    'Y': 'Y (test func)',
                    'Delta_Y': 'Delta Y (test func)',
                    'No_Ideal_Func': 'No. of ideal func',
                     'Y_Ideal': 'Y_Ideal',
                    'Mapping_Threshold': 'Mapping_Threshold',
                    'Original_Train_Func': 'Original_Train_Func'
                })

                self.db_manager.write_test_results(results_df, if_exists='replace')

        except FileNotFoundError:
            print(f"Error: Test file not found at {test_file_path}")
            raise
        except Exception as e:
            print(f"Error processing test file: {e}")
            raise

    def map_point(self, x_test: float, y_test: float):
        """
        Applies the mapping for a single (x, y) test point.
        """
        try:
            ideal_y_values = self.ideal_data.loc[x_test]
        except KeyError:
            return

        best_match = None
        min_deviation = float('inf')

        for ideal_col in self.chosen_ideal_cols:

            y_ideal = ideal_y_values[ideal_col]
            threshold = self.max_deviations[ideal_col] * self.sqrt_2_factor
            deviation = abs(y_test - y_ideal)
            if deviation <= threshold:
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_match = ideal_col

        if best_match is not None:
            winning_y_ideal = ideal_y_values[best_match]
            winning_threshold = self.max_deviations[best_match] * self.sqrt_2_factor
            train_func = self.ideal_to_train_map[best_match]
            self.results.append({
                'X': x_test,
                'Y': y_test,
                'Delta_Y': min_deviation,
                'No_Ideal_Func': best_match,
                'Y_Ideal': winning_y_ideal,
                'Mapping_Threshold': winning_threshold,
                'Original_Train_Func': train_func
            })