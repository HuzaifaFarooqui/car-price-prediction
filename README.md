# Car Price Prediction

A Python machine-learning project that cleans a used-vehicle dataset, engineers price-related features, and predicts car selling prices.

## Dataset

`car data.csv` contains 301 records, including cars, motorcycles, and scooters. `Selling_Price` is the prediction target; prices are in lakhs. Available fields include vehicle name, year, present price, driven kilometers, fuel type, seller type, transmission, and owner count. The dataset does not include horsepower or a separate brand field.

## Setup and run

Python 3.10 or newer is recommended. In PowerShell, from this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python car_price_prediction.py
```

The script removes duplicate records and known two-wheelers, creates the `Vehicle_Age` and `Usage_Rate` features, and saves the trained model and actual-versus-predicted chart in `outputs/`. By default, it prints just one line with the hold-out R², MAE, and RMSE. Add `--details` to print the data summary and preprocessing diagnostics.

## Selected model and evaluation

The final program uses one model: `RandomForestRegressor` with 200 trees. In model selection, it performed best among the tested random forest, extra trees, and gradient boosting candidates. Repeated five-fold cross-validation on the cleaned car rows gave mean R² **0.869** (fold standard deviation **0.062**) and mean MAE **0.926 lakhs**. With the script's fixed 60/40 split, the hold-out results were R² **0.793**, MAE **1.152 lakhs**, and RMSE **2.262 lakhs**. Scores may vary with other splits because this dataset is small.

## Feature engineering and leakage prevention

- **Sanitation:** a curated, normalized model-name catalog removes known motorcycles and scooters before imputation or scaling. The catalog is specific to this dataset. Low price and `Owner == 0` are diagnostic indicators, not classification rules.
- **Vehicle age:** current year minus model year.
- **Usage rate:** driven kilometers divided by vehicle age plus one.
- **Categories:** car name, fuel type, seller type, and transmission are one-hot encoded.
- **Numeric preprocessing:** median imputation and standard scaling are fitted inside the model pipeline using training rows only.
- **Depreciation ratio:** calculated for analysis, but excluded from model inputs because it uses `Selling_Price`, the target. Including it would leak the answer.

## Learning guide

1. Read the EDA and vehicle-indicator functions, then compare their summaries with the CSV.
2. Review `TWO_WHEELER_MODELS` and verify the removed records are motorcycles or scooters.
3. Trace `engineer_features` and explain why the depreciation ratio cannot be used as a predictor.
4. Compare the hold-out MAE, RMSE, and R² with the cross-validation results. R² can change substantially across splits.
5. Run `python car_price_prediction.py --details` to see the exploratory data and sanitation diagnostics. Open `outputs/actual_vs_predicted.png`; points closer to the diagonal are more accurate.

## Limitations

The dataset is small and reflects an older market. Vehicle age uses the system's current year, which may not match when these listings were recorded. Better real-world predictions need current listings and fields such as brand, horsepower, condition, location, and service history.

For submission, include this README, `car_price_prediction.py`, `car data.csv`, `requirements.txt`, and `.gitignore`. Exclude `.venv`, `__pycache__`, and generated `outputs`; the script recreates outputs when run.
