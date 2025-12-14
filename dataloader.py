import pandas as pd

class DataLoader:
    """
    class for loading data.
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Loads data from a file. This method should be overridden."""
        print(f"Loading data from {self.file_path}")
        try:
            df = pd.read_csv(self.file_path)
            if 'x' in df.columns:
                df.rename(columns={'x': 'X'}, inplace=True)
            if 'X' not in df.columns:
                raise ValueError("No 'X' column found in the data.")

            print(f"Data loaded from {self.file_path}")
            return df
        except FileNotFoundError as e:
            print(f"Error: File not found at {e.filename}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise


class TrainingLoader(DataLoader):
    """Loads and processes the training datasets."""

    def __init__(self, file_path: str):
        # file_paths_list should be a list of the 4 training CSV paths
        super().__init__(file_path)

    def load_data(self):
        """
        Loads the CSVs and merges them into a single DataFrame matching Table 1.
        """
        df = super().load_data()
        print("Training data loaded...")
        return df


class IdealFunctionLoader(DataLoader):
    """Loads  ideal functions."""

    def __init__(self, file_path):
        super().__init__(file_path)

    def load_data(self):
        """
        Loads the ideal functions CSV into a DataFrame
        """
        df = super().load_data()
        print("Ideal functions loaded.")
        return df