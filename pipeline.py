import traceback

from analyzer import Analyzer
from database import DatabaseManager
from exceptions.data_mismatch_error import DataMismatchError
from processor import DataMapper
from dataloader import TrainingLoader, IdealFunctionLoader
from visualizer import Visualizer

""" Pipeline that runs all stages of the ideal functions analysis in order.

Stages: load CSV files, write them to the SQlite database, select best-fit ideal functions, map the test points and create the plots

"""
class Pipeline:
    """
    Run the full analysis from the input CSV files to the output files
    """
    def __init__(self, train_path, ideal_path, test_path, output_csv_path, db_name="idealfunction.db"):
        """
        Store the file paths used by the pipeline
        :param train_path: path to the training data CSV
        :param ideal_path: path to the ideal functions CSV
        :param test_path: path to the test data CSV
        :param output_csv_path: path for the mapped test results CSV
        :param db_name: name of the SQLite database file
        """
        self.train_path = train_path
        self.ideal_path = ideal_path
        self.test_path = test_path
        self.output_csv_path = output_csv_path
        self.db_name = db_name

    def run(self):
        """
        Run all pipeline stages in order
        Errors are caught and printed, so the program ends with a message instead of a traceback
        """
        try:
            db_manager = DatabaseManager(db_name=self.db_name)

            train_loader = TrainingLoader(self.train_path)
            ideal_loader = IdealFunctionLoader(self.ideal_path)
            training_df = train_loader.load_data()
            ideal_df = ideal_loader.load_data()

            db_manager.write_data_with_x_index(
                training_df, 'training_data', if_exists='replace'
            )
            db_manager.write_data_with_x_index(
                ideal_df, 'ideal_functions', if_exists='replace'
            )

            analyzer = Analyzer(db_manager)
            best_fit_ranking, max_deviations = analyzer.run_analysis()
            analyzer.export_summary_table()
            print("\n---- Analysis results----")
            print(f"Best-fit function ranking: {best_fit_ranking}")
            print(f"Max deviations: {max_deviations}")

            processor = DataMapper(
                db_manager=db_manager,
                ideal_df=ideal_df,
                max_deviations=max_deviations,
                best_fit_ranking=best_fit_ranking,
            )
            mapped_results = processor.process_test_file(self.test_path, self.output_csv_path)
            processor.compute_mapping_statistics()
            processor.compute_rejection_summary()

            visualizer = Visualizer(db_manager, best_fit_ranking, mapped_results)
            visualizer.generate_and_save_plots(self.test_path)

        except DataMismatchError as e:
            print(f"Data mismatch error: {e}")
        except FileNotFoundError as e:
            print(f"Error: File not found at {e.filename}")
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
