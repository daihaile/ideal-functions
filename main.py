import sqlalchemy

from analyzer import Analyzer
from database import DatabaseManager
from exceptions.DataMismatcherror import DataMismatchError
from processor import DataMapper
from dataloader import TrainingLoader, IdealFunctionLoader, DataLoader
from visualizer import Visualizer

if __name__ == "__main__":
    try:

        TRAIN_FILE_PATH = 'data/train.csv'
        IDEAL_FILE_PATH = 'data/ideal.csv'
        TEST_FILE_PATH = 'data/test.csv'

        OUTPUT_FILE_PATH = 'output/test_results.csv'

        db_manager = DatabaseManager(db_name="idealfunction.db")

        # Loading data from CSV files into DataFrames
        train_loader = TrainingLoader(TRAIN_FILE_PATH)
        ideal_loader = IdealFunctionLoader(IDEAL_FILE_PATH)
        test_loader = DataLoader(TEST_FILE_PATH)

        training_df = train_loader.load_data()
        ideal_df = ideal_loader.load_data()

        db_manager.write_data_with_x_index(training_df, 'training_data', if_exists='replace')
        db_manager.write_data_with_x_index(ideal_df, 'ideal_functions', if_exists='replace')

        # Loading end

        # Analysis start

        analyzer = Analyzer(db_manager)
        best_fit_ranking, max_deviations = analyzer.run_analysis()

        print("\n---- Analysis results----")
        print(f"Best-fit function ranking: {best_fit_ranking}")
        print(f"Max deviations: {max_deviations}")


        # Analysis end

        # data processing

        processor = DataMapper(
            db_manager=db_manager,
            ideal_df=ideal_df,
            max_deviations=max_deviations,
            best_fit_ranking=best_fit_ranking
        )

        processor.process_test_file(TEST_FILE_PATH, OUTPUT_FILE_PATH)
        # data processing end

        # visualization

        visualizer = Visualizer(db_manager, best_fit_ranking)
        visualizer.generate_and_save_plots(TEST_FILE_PATH)

        # visualization end
    except DataMismatchError as e:
        print(f"Data mismatch error: {e}")
    except FileNotFoundError as e:
        print(f"Error: File not found at {e.filename}")
    except Exception as e:
        print(f"Error: {e}")













