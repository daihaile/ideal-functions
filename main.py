from pipeline import Pipeline

if __name__ == "__main__":
    Pipeline(
        train_path='data/train.csv',
        ideal_path='data/ideal.csv',
        test_path='data/test.csv',
        output_csv_path='output/test_results.csv',
    ).run()
