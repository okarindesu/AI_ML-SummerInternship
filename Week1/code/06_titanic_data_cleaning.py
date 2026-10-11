"""Mini Project 1: Titanic Survival Prediction – Data Cleaning Project.

Expected input: Kaggle Titanic train.csv. The original dataset is not bundled.
Tasks: inspect data, clean missing values, encode Sex and Embarked, plot age
distribution, and export a cleaned CSV. This script performs data cleaning; it does
not train a survival classifier because the supplied brief asks for cleaning only.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="train.csv",
                        help="Path to Kaggle Titanic train.csv")
    parser.add_argument("--output", default="titanic_cleaned.csv")
    parser.add_argument("--plot", default="age_distribution.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    print("=== Initial overview ===")
    print("Shape:", df.shape)
    print(df.head())
    print("\n=== Column information ===")
    df.info()
    print("\n=== Missing values before cleaning ===")
    print(df.isna().sum().sort_values(ascending=False))
    print("\n=== Summary statistics ===")
    print(df.describe(include="all").T)

    # Plot age distribution before imputation so the observed distribution remains visible.
    if "Age" in df.columns:
        age_values = pd.to_numeric(df["Age"], errors="coerce").dropna()
        plt.figure(figsize=(8, 5))
        plt.hist(age_values, bins=25, edgecolor="black")
        plt.title("Titanic Passenger Age Distribution (observed values)")
        plt.xlabel("Age")
        plt.ylabel("Number of passengers")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        plt.close()
        print(f"\nSaved age plot: {args.plot}")

    # Numeric imputation: median for Age; other numeric columns are left unchanged
    # unless they contain missing values, in which case their median is used.
    for col in df.select_dtypes(include="number").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill Embarked with the most frequent value; other text columns use a
    # missing-value label rather than inventing a category.
    if "Embarked" in df.columns:
        mode = df["Embarked"].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        df["Embarked"] = df["Embarked"].fillna(fill_value)

    for col in df.select_dtypes(include="object").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    # Encode Sex as a binary column when present; keep it interpretable as 0/1.
    if "Sex" in df.columns:
        sex_map = {"female": 0, "male": 1}
        df["Sex"] = df["Sex"].astype(str).str.lower().map(sex_map).fillna(-1).astype(int)

    # One-hot encode Embarked, retaining category-specific indicator columns.
    if "Embarked" in df.columns:
        df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked", dtype=int)

    df.to_csv(args.output, index=False)
    print("\n=== Missing values after cleaning ===")
    print(df.isna().sum().sort_values(ascending=False))
    print(f"\nCleaned data saved to: {args.output}")
    print("Note: this is a cleaned dataset, not a trained prediction model.")

if __name__ == "__main__":
    main()
