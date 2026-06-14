import traceback

from analyzer import Analyzer
from database import DatabaseManager
from exceptions.DataMismatcherror import DataMismatchError
from processor import DataMapper
from dataloader import TrainingLoader, IdealFunctionLoader
from visualizer import Visualizer


class Pipeline:
    def __init__(self, train_path, ideal_path, test_path, output_csv_path, db_name="idealfunction.db"):
        self.train_path = train_path
        self.ideal_path = ideal_path
        self.test_path = test_path
        self.output_csv_path = output_csv_path
        self.db_name = db_name

    def run(self):
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
            print("\n---- Analysis results----")
            print(f"Best-fit function ranking: {best_fit_ranking}")
            print(f"Max deviations: {max_deviations}")

            processor = DataMapper(
                db_manager=db_manager,
                ideal_df=ideal_df,
                max_deviations=max_deviations,
                best_fit_ranking=best_fit_ranking,
            )
            processor.process_test_file(self.test_path, self.output_csv_path)

            visualizer = Visualizer(db_manager, best_fit_ranking)
            visualizer.generate_and_save_plots(self.test_path)

        except DataMismatchError as e:
            print(f"Data mismatch error: {e}")
        except FileNotFoundError as e:
            print(f"Error: File not found at {e.filename}")
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
