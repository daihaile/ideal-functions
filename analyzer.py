import pandas as pd
import numpy as np
from database import DatabaseManager

class Analyzer:
    """
    handles analysis of finding best fit functions and calculating deviation
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initializes the analyzer by loading data from database
        :param db_manager:
        """
        self.db_manager = db_manager

        #load data from db

        self.train_df = self.db_manager.read_table_to_dataframe("training_data")
        self.ideal_df = self.db_manager.read_table_to_dataframe("ideal_functions")

        self.train_df_aligned, self.ideal_df_aligned = self.train_df.align(self.ideal_df, join='inner', axis=0, copy=True)

        self.best_fit_map = {}
        self.max_deviations = {}

    def run_analysis(self):
        """
        runs analysis to find best fit functions + deviation thresholds
        :return:
        """

        print("Start analysis")

        training_cols = [col for col in self.train_df_aligned.columns if col.startswith('y')]
        ideal_cols = [col for col in self.ideal_df_aligned.columns if col.startswith('y')]

        if not training_cols or not ideal_cols:
            print("Error: No training or ideal columns found. Check data and column names.")
            return {}, {}

        for train_col in training_cols:
            print(f" Analyzing {train_col}...")
            best_ideal_col = None
            min_sum_of_squares = float('inf')
            train_series = self.train_df_aligned[train_col]

            for ideal_col in ideal_cols:
                ideal_series = self.ideal_df_aligned[ideal_col]
                sum_of_squares = ((train_series - ideal_series) ** 2).sum()

                if sum_of_squares < min_sum_of_squares:
                    min_sum_of_squares = sum_of_squares
                    best_ideal_col = ideal_col

            if best_ideal_col:
                self.best_fit_map[train_col] = best_ideal_col
                print(f"Best fit function for {train_col} is {best_ideal_col}")

                best_ideal_series = self.ideal_df_aligned[best_ideal_col]
                max_deviation = (abs(train_series - best_ideal_series)).max()
                self.max_deviations[best_ideal_col] = max_deviation

                print(f"Max deviation for {best_ideal_col} is {max_deviation:.4f}")
            else:
                print(f"Error: No best fit found for {train_col}")



        print("Analysis finished")
        return self.best_fit_map, self.max_deviations