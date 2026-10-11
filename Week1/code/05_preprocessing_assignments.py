"""Week 1 Assignments 2 and 3: missing-value imputation and categorical encoding.

This example reads a CSV, imputes numeric columns with their median and categorical
columns with their most frequent value, then encodes categorical features.
It saves the transformed dataset to preprocessed_output.csv.

For a real predictive ML pipeline, fit preprocessing only on training data (e.g. with
sklearn Pipeline/ColumnTransformer) to avoid data leakage.
"""
import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to input CSV")
    parser.add_argument("--output", default="preprocessed_output.csv")
    parser.add_argument("--label-column", default=None,
                        help="Optional one column to LabelEncode; other categorical columns use one-hot encoding")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    print("Before preprocessing:")
    print("Shape:", df.shape)
    print("Missing values:\n", df.isna().sum())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    if args.label_column:
        if args.label_column not in categorical_cols:
            raise ValueError(f"--label-column must name a categorical column. Categorical columns: {categorical_cols}")
        encoder = LabelEncoder()
        df[args.label_column] = encoder.fit_transform(df[args.label_column].astype(str))
        categorical_cols.remove(args.label_column)

    # One-hot encode remaining categorical columns; drop_first=False keeps all categories explicit.
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

    df.to_csv(args.output, index=False)
    print("\nAfter preprocessing:")
    print("Shape:", df.shape)
    print("Missing values remaining:", int(df.isna().sum().sum()))
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
