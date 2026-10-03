"""Car price prediction with EDA, two-wheeler sanitation, and safe features."""

import argparse
from datetime import date
from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "car data.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "car_price_model.joblib"
TARGET = "Selling_Price"
CURRENT_YEAR = date.today().year

# Curated model-name profile for this dataset. Owner count and price are not
# reliable vehicle-type labels: almost every row has Owner == 0, and the price
# ranges of small cars and expensive motorcycles overlap.
TWO_WHEELER_MODELS = {
    "bajaj discover 125", "bajaj ct 100", "hero cbz xtreme", "hero ignitor disc",
    "hero hunk", "hero super splendor", "honda cb twister", "honda cb shine",
    "bajaj pulsar 150", "tvs wego", "suzuki access 125", "hero glamour",
    "bajaj discover 100", "hero splender plus", "hero honda passion pro",
    "activa 3g", "honda karizma", "honda activa 125", "tvs jupyter",
    "hero honda cbz extreme", "honda cb unicorn", "activa 4g", "bajaj pulsar 135 ls",
    "yamaha fz s", "honda cb trigger", "hero passion pro", "hero splender ismart",
    "bajaj avenger street 220", "honda activa 4g", "tvs apache rtr 180", "tvs sport",
    "honda dream yuga", "yamaha fazer", "bajaj pulsar ns 200", "hero passion x pro",
    "bajaj pulsar 220 f", "bajaj avenger 220 dtsi", "yamaha fz s v 2.0",
    "bajaj avenger 150 street", "yamaha fz v 2.0", "bajaj pulsar ns 200",
    "tvs apache rtr 160", "hero extreme", "honda cbr 150", "honda cb hornet 160r",
    "bajaj avenger 220", "bajaj avenger 150", "yamaha fz 16",
    "royal enfield classic 500", "royal enfield bullet 350", "bajaj pulsar rs200",
    "royal enfield thunder 500", "royal enfield classic 350", "mahindra mojo xt300",
    "ktm 390 duke", "royal enfield thunder 350", "ktm rc390", "ktm rc200",
    "bajaj dominar 400", "um renegade mojave", "hyosung gt250r",
}

NUMERIC_FEATURES = ["Vehicle_Age", "Present_Price", "Driven_kms", "Owner", "Usage_Rate"]
CATEGORICAL_FEATURES = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]


def normalize_name(value: object) -> str:
    """Normalize a model name so case and repeated whitespace do not matter."""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def inspect_distributions(data: pd.DataFrame) -> None:
    """Print a compact distribution summary for key numeric measures."""
    print("\nDataset overview")
    summary = []
    for column in ["Present_Price", TARGET, "Driven_kms"]:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        summary.append(
            {
                "Feature": column,
                "Count": values.size,
                "Mean": values.mean(),
                "Median": values.median(),
                "Min": values.min(),
                "Max": values.max(),
            }
        )
    print(pd.DataFrame(summary).to_string(index=False, formatters={
        "Mean": "{:,.2f}".format, "Median": "{:,.2f}".format,
        "Min": "{:,.2f}".format, "Max": "{:,.2f}".format,
    }))


def report_vehicle_indicators(data: pd.DataFrame) -> None:
    """Report name-profile and low-price indicators of possible two-wheelers."""
    names = data["Car_Name"].map(normalize_name)
    profile = names.isin(TWO_WHEELER_MODELS)
    low_price = (data["Present_Price"] <= 3.5) & (data[TARGET] <= 1.5)
    print(
        "Vehicle-type indicators: "
        f"{profile.sum()} known two-wheeler rows; {low_price.sum()} also match the low-price screen. "
        f"Owner=0 occurs in {(data['Owner'] == 0).sum()}/{len(data)} rows and is not used to classify bikes."
    )


def remove_non_car_rows(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Remove catalogued motorcycle/scooter rows before imputation or scaling.

    The vehicle profile uses model names, not price or ownership. Filtering
    these rows first avoids mixing motorcycle and car distributions when later
    imputers and scalers learn medians, means, and variances. This label-based
    sanitation is explainable for this known dataset; it should be reviewed if
    a new CSV introduces models absent from the curated profile.
    """
    normalized = data["Car_Name"].map(normalize_name)
    bike_mask = normalized.isin(TWO_WHEELER_MODELS)
    clean = data.loc[~bike_mask].copy()
    if verbose:
        print(f"Removed {int(bike_mask.sum())} two-wheeler rows; retained {len(clean)} car rows.")
    if clean.empty:
        raise ValueError("Sanitation removed every row; check the two-wheeler model profile.")
    return clean


def engineer_features(
    data: pd.DataFrame, verbose: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Build requested domain features and return predictors separately from target.

    Depreciation_Ratio is calculated for the requested domain analysis, but
    excluded from predictors because it contains Selling_Price, the answer the
    model is meant to predict. Including it would leak the target into training.
    """
    engineered = data.copy()
    for column in ["Year", "Present_Price", "Driven_kms", "Owner", TARGET]:
        engineered[column] = pd.to_numeric(engineered[column], errors="coerce")
    engineered = engineered.dropna(subset=[TARGET, "Year"])
    engineered = engineered.loc[engineered[TARGET] >= 0].copy()
    engineered["Vehicle_Age"] = (CURRENT_YEAR - engineered["Year"]).clip(lower=0)
    valid_present_price = engineered["Present_Price"].where(engineered["Present_Price"] > 0)
    engineered["Depreciation_Ratio"] = engineered[TARGET] / valid_present_price
    engineered["Usage_Rate"] = engineered["Driven_kms"] / (engineered["Vehicle_Age"] + 1)
    if verbose:
        print(
            f"Created Vehicle_Age (reference year {CURRENT_YEAR}) and Usage_Rate. "
            "Depreciation_Ratio is diagnostic only because it uses the target."
        )

    predictors = engineered[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    target = engineered[TARGET].copy()
    return predictors, target


def make_preprocessor() -> ColumnTransformer:
    """Create imputation, one-hot encoding, and numeric scaling in a pipeline."""
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_model() -> Pipeline:
    """Build the selected random-forest model with preprocessing in one pipeline."""
    return Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=1,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(y_true: pd.Series, predictions) -> dict[str, float]:
    """Return familiar held-out regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": mean_squared_error(y_true, predictions) ** 0.5,
        "R2": r2_score(y_true, predictions),
    }


def main() -> None:
    """Run diagnostics, sanitation, feature creation, model evaluation, and save."""
    parser = argparse.ArgumentParser(description="Predict used-car selling prices.")
    parser.add_argument(
        "--details", action="store_true", help="Print dataset and preprocessing diagnostics."
    )
    args = parser.parse_args()
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset: {DATA_PATH}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw_data = pd.read_csv(DATA_PATH).drop_duplicates().copy()
    required = {
        "Car_Name", "Year", "Present_Price", TARGET, "Driven_kms", "Owner",
        "Fuel_Type", "Selling_type", "Transmission",
    }
    missing = required.difference(raw_data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    for column in ["Year", "Present_Price", TARGET, "Driven_kms", "Owner"]:
        raw_data[column] = pd.to_numeric(raw_data[column], errors="coerce")
    raw_data = raw_data.dropna(subset=[TARGET, "Present_Price", "Driven_kms", "Owner"])
    if args.details:
        inspect_distributions(raw_data)
        report_vehicle_indicators(raw_data)

    cars = remove_non_car_rows(raw_data, verbose=args.details)
    X, y = engineer_features(cars, verbose=args.details)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    model = make_model()
    model.fit(X_train, y_train)
    held_out_predictions = model.predict(X_test)
    scores = evaluate(y_test, held_out_predictions)
    print(
        f"Random Forest (test n={len(X_test)}) | "
        f"R2={scores['R2']:.3f} | MAE={scores['MAE']:.3f}  | "
        f"RMSE={scores['RMSE']:.3f} "
    )

    # Save a final model fitted on all clean car rows after held-out evaluation.
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, held_out_predictions, alpha=0.75, edgecolor="white")
    bounds = [min(y_test.min(), held_out_predictions.min()), max(y_test.max(), held_out_predictions.max())]
    ax.plot(bounds, bounds, "--", color="darkorange", label="Perfect prediction")
    ax.set(title="Random forest: actual vs predicted", xlabel="Actual selling price (lakhs)", ylabel="Predicted selling price (lakhs)")
    ax.legend()
    fig.tight_layout()
    chart_path = OUTPUT_DIR / "actual_vs_predicted.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
