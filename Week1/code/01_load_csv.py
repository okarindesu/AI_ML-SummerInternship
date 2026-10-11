"""Practice 1: Load a CSV with pandas and print the first 10 rows."""
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to a CSV file")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    print("First 10 rows:")
    print(df.head(10))
    print("\nShape:", df.shape)
    print("\nColumns:", list(df.columns))

if __name__ == "__main__":
    main()
