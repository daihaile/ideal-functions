import pandas as pd

from exceptions.DataMismatcherror import DataMismatchError


class DataLoader:
    """
    class for loading data. Subclasses can reuse load_data() function to load different types of data.
    """

    def __init__(self, file_path):
        """Create a loader for a giving csv file path"""
        self.file_path = file_path

    def load_data(self):
        """Read csv file and return a DataFrame
         :return
            pandas DataFrame
        :raises
            DataMismatchError

         """
        print(f"Loading data from {self.file_path}")
        try:
            df = pd.read_csv(self.file_path)
            if 'x' in df.columns:
                df.rename(columns={'x': 'X'}, inplace=True)
            if 'X' not in df.columns:
                raise DataMismatchError(f"Missing required column 'X' in '{self.file_path}'. Found columns: {list(df.columns)}")

            print(f"Data loaded from {self.file_path}")
            return df
        except FileNotFoundError as e:
            print(f"Error: File not found at {e.filename}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise


class TrainingLoader(DataLoader):
    """Loads the training dataset."""

    def __init__(self, file_path: str):
        """Create a loader for the training CSV file."""
        super().__init__(file_path)

    def load_data(self):
        """
        Load and returns the training data
        """
        df = super().load_data()
        print("Training data loaded...")
        return df


class IdealFunctionLoader(DataLoader):
    """Loads ideal functions."""

    def __init__(self, file_path):
        """Create a loader for the ideal functions CSV file."""
        super().__init__(file_path)

    def load_data(self):
        """
        Loads the ideal functions CSV into a DataFrame
        """
        df = super().load_data()
        print("Ideal functions loaded.")
        return df