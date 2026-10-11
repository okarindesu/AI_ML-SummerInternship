"""Practice 3 and 4: Fit linear regression and predict house price from area.
Expected CSV columns by default: area, price. Override with --area-col and --price-col.
R² is a regression score; it is not classification accuracy.
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="house_prices_sample.csv")
    parser.add_argument("--area-col", default="area")
    parser.add_argument("--price-col", default="price")
    parser.add_argument("--area", type=float, default=None, help="Area to predict for (same unit as dataset)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    for col in (args.area_col, args.price_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Available: {list(df.columns)}")

    data = df[[args.area_col, args.price_col]].dropna()
    if len(data) < 3:
        raise ValueError("Need at least 3 complete rows to train and evaluate this example.")

    X = data[[args.area_col]]
    y = data[args.price_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Intercept: {model.intercept_:.3f}")
    print(f"Coefficient for {args.area_col}: {model.coef_[0]:.3f}")
    print(f"R²: {r2_score(y_test, predictions):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.3f}")
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE: {rmse:.3f}")

    area = args.area
    if area is None:
        try:
            area = float(input(f"Enter house area ({args.area_col}): "))
        except ValueError:
            raise SystemExit("Please enter a numeric area.")
    predicted_price = model.predict(pd.DataFrame({args.area_col: [area]}))[0]
    print(f"Predicted price for area {area:g}: {predicted_price:.2f}")

if __name__ == "__main__":
    main()
