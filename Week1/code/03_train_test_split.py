"""Practice 2: Split a CSV into training and test sets using scikit-learn.
Writes train.csv and test.csv. Set --target to use a stratified split for classification.
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to a CSV file")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--target", default=None, help="Optional classification target for stratification")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-prefix", default="split")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    stratify = None
    if args.target:
        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found. Columns: {list(df.columns)}")
        stratify = df[args.target]

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=stratify
    )
    train_path = f"{args.output_prefix}_train.csv"
    test_path = f"{args.output_prefix}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Rows: total={len(df)}, train={len(train_df)}, test={len(test_df)}")
    print(f"Saved: {train_path}, {test_path}")

if __name__ == "__main__":
    main()
