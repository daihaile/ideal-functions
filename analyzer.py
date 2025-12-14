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

        # load data from db

        self.train_df = self.db_manager.read_table_to_dataframe("training_data")
        self.ideal_df = self.db_manager.read_table_to_dataframe("ideal_functions")

        self.train_df_aligned, self.ideal_df_aligned = self.train_df.align(self.ideal_df, join='inner', axis=0,
                                                                           copy=True)

        self.best_fit_ranking = {}
        self.max_deviations = {}

    def run_analysis(self):
        """
        runs analysis to find top 3 best fit functions + deviation thresholds
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
            results_list = []
            train_series = self.train_df_aligned[train_col]

            for ideal_col in ideal_cols:
                ideal_series = self.ideal_df_aligned[ideal_col]
                sum_of_squares = ((train_series - ideal_series) ** 2).sum()

                results_list.append((sum_of_squares, ideal_col))

            sorted_results = sorted(results_list)
            best_fit_sum, best_ideal_col = sorted_results[0]

            self.best_fit_ranking[train_col] = [
                (col, ssq) for ssq, col in sorted_results[:3] # top 3
            ]
            print(f"    -> Best fit for {train_col} is {best_ideal_col} (Sum of squares: {best_fit_sum:.2f})")

            best_ideal_series = self.ideal_df_aligned[best_ideal_col]
            max_deviation = (abs(train_series - best_ideal_series)).max()
            self.max_deviations[best_ideal_col] = max_deviation
            print(f"    -> Max deviation for {best_ideal_col} is {max_deviation:.4f}")

        print("Analysis finished")
        return self.best_fit_ranking, self.max_deviations
