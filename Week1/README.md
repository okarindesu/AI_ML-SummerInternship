# AI + ML Week 1 Study Pack

This folder contains a summary of the supplied AI notes, code for every listed Week 1 practice/assignment task, and a Titanic data-cleaning mini-project.

## Files
- `AI_ML_Notes_Summary.pdf` — concise revision summary of the 211-page AI notes and Week 1 ML documents.
- `code/01_load_csv.py` — practice question: load a CSV and print the first 10 rows.
- `code/02_explore_dataset.py` — Week 1 assignment: `.info()`, `.describe()`, and missing-value counts.
- `code/03_train_test_split.py` — practice question: split a CSV into train/test files.
- `code/04_linear_regression_house_price.py` — practice questions: fit/evaluate linear regression and predict house price from area.
- `code/05_preprocessing_assignments.py` — Week 1 assignments: median/mode imputation, LabelEncoder option, and one-hot encoding.
- `code/06_titanic_data_cleaning.py` — mini project: clean Titanic data, encode `Sex` and `Embarked`, plot age distribution, export CSV.
- `code/house_prices_sample.csv` — small illustrative dataset to test the regression script. It is synthetic and must not be treated as real market data.
- `source_materials/` — copies of the three PDFs you supplied.

## Setup
Use Python 3.10+ if possible.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the practice code
Run commands from the `AI_ML_Study_Pack` folder:

```bash
python code/01_load_csv.py your_dataset.csv
python code/02_explore_dataset.py your_dataset.csv
python code/03_train_test_split.py your_dataset.csv
python code/03_train_test_split.py your_dataset.csv --target Survived
python code/04_linear_regression_house_price.py code/house_prices_sample.csv --area 1500
python code/05_preprocessing_assignments.py your_dataset.csv
python code/05_preprocessing_assignments.py your_dataset.csv --label-column Category
```

## Run the Titanic mini-project
1. Download the Titanic `train.csv` file from the Kaggle Titanic competition. The dataset is not included in the uploaded notes.
2. Put `train.csv` in this folder, or pass its path:
   ```bash
   python code/06_titanic_data_cleaning.py path/to/train.csv
   ```
3. The script creates:
   - `titanic_cleaned.csv`
   - `age_distribution.png`

## Important notes
- The project brief asks for data cleaning and export; it does not explicitly require training a survival prediction model, so this script does not train one.
- `R²`, MAE and RMSE are regression metrics. R² is not classification accuracy.
- The sample house-price data is synthetic for checking that the script runs.
- The generic preprocessing script demonstrates the concepts on a whole CSV for learning. For a real predictive workflow, fit imputation/encoding/scaling only on training data using a scikit-learn `Pipeline`/`ColumnTransformer` to avoid data leakage.
- The `LabelEncoder` option is included because the assignment explicitly names it. For nominal input features, one-hot encoding is often more appropriate because integer labels may imply an order that does not exist.
