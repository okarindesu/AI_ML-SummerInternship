"""Week 1 Assignment 1: Explore a dataset with info() and describe()."""
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to a CSV file")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)

    print("=== First 10 rows ===")
    print(df.head(10))
    print("\n=== Dataset shape ===")
    print(df.shape)
    print("\n=== Column information ===")
    df.info()
    print("\n=== Descriptive statistics (all columns) ===")
    print(df.describe(include="all").T)
    print("\n=== Missing values per column ===")
    print(df.isna().sum())

if __name__ == "__main__":
    main()
