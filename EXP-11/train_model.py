"""Train price and price-band models from Craigslist's vehicles.csv dataset.

Run ``python train_model.py``. If data/vehicles.csv is absent, the script uses
Kaggle's Craigslist Cars and Trucks Data download endpoint automatically. It
accepts either a ZIP download (Kaggle's normal response) or a direct CSV URL.
"""
import argparse
import pickle
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/austinreese/craigslist-carstrucks-data"
PRICE_MIN, PRICE_MAX = 200, 200_000
NUMERIC_COLUMNS = ["year", "odometer"]
CATEGORICAL_COLUMNS = [
    "manufacturer", "model", "condition", "cylinders", "fuel",
    "title_status", "transmission", "drive", "type", "paint_color", "state",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Craigslist vehicle models.")
    parser.add_argument("--csv", default="data/vehicles.csv", help="Path to vehicles.csv")
    parser.add_argument("--download-url", default=DATASET_URL,
                        help="Dataset ZIP or CSV URL used if --csv is absent")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--max-rows", type=int, default=500,
                        help="Rows used for training (default: 500, for a quick demo)")
    parser.add_argument("--full-data", action="store_true",
                        help="Use every valid row; this takes substantially longer")
    return parser.parse_args()


def get_csv(path, download_url):
    path = Path(path)
    if not path.exists() and download_url:
        path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_file = path.with_suffix(".download")
        print(f"Downloading dataset from {download_url} ...")
        urlretrieve(download_url, downloaded_file)
        if zipfile.is_zipfile(downloaded_file):
            with zipfile.ZipFile(downloaded_file) as archive:
                member = next((item for item in archive.namelist()
                               if Path(item).name.lower() == "vehicles.csv"), None)
                if member is None:
                    raise FileNotFoundError("The downloaded ZIP does not contain vehicles.csv.")
                with archive.open(member) as source, open(path, "wb") as destination:
                    shutil.copyfileobj(source, destination)
            downloaded_file.unlink()
        else:
            downloaded_file.replace(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}. Place vehicles.csv there, or pass --csv / --download-url."
        )
    return path


def make_preprocessor(numeric, categorical):
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Rare values are grouped during fit, preventing thousands of one-hot columns.
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")


def price_band(price):
    """Turn price into the classification target, after feature selection."""
    return pd.cut(price, bins=[-np.inf, 10_000, 25_000, np.inf],
                  labels=["budget", "mid-range", "premium"])


def report_regression(name, model, x_test, y_test):
    prediction = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    print(f"{name}: MAE={mean_absolute_error(y_test, prediction):.2f}, "
          f"RMSE={rmse:.2f}, R2={r2_score(y_test, prediction):.4f}")


def main():
    args = parse_args()
    csv_path = get_csv(args.csv, args.download_url)
    # Read only the target and fields used by the models, not the large text/image columns.
    required = {"price", *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS}
    frame = pd.read_csv(csv_path, usecols=lambda column: column in required, low_memory=False)
    if "price" not in frame:
        raise ValueError("The CSV must contain a 'price' column.")

    # Price creates the targets only. It is deliberately not part of `features`.
    available_numeric = [c for c in NUMERIC_COLUMNS if c in frame.columns]
    available_categorical = [c for c in CATEGORICAL_COLUMNS if c in frame.columns]
    feature_columns = available_numeric + available_categorical
    if not feature_columns:
        raise ValueError("None of the expected vehicle feature columns were found.")

    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.loc[frame["price"].between(PRICE_MIN, PRICE_MAX)].copy()
    frame["price_band"] = price_band(frame["price"])
    frame = frame.dropna(subset=["price_band"])
    sample_size = None if args.full_data else args.max_rows
    if sample_size and len(frame) > sample_size:
        frame = frame.sample(sample_size, random_state=RANDOM_STATE)
    if len(frame) < 20:
        raise ValueError("Too few valid rows remain after price filtering.")

    features = frame[feature_columns].copy()
    y_price = frame["price"].astype(float)
    y_band = frame["price_band"].astype(str)
    # Both tasks use the exact same held-out rows. Stratification preserves bands.
    train_idx, test_idx = train_test_split(
        frame.index, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y_band
    )
    x_train, x_test = features.loc[train_idx], features.loc[test_idx]
    price_train, price_test = y_price.loc[train_idx], y_price.loc[test_idx]
    band_train, band_test = y_band.loc[train_idx], y_band.loc[test_idx]

    classifiers = {
        "decision_tree_classifier": DecisionTreeClassifier(random_state=RANDOM_STATE, min_samples_leaf=3),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=100, max_depth=25, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2
        ),
    }
    regressors = {
        "decision_tree_regressor": DecisionTreeRegressor(random_state=RANDOM_STATE, min_samples_leaf=3),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=100, max_depth=25, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2
        ),
        "linear_regression": LinearRegression(),
    }
    trained_classifiers, trained_regressors, classifier_metrics = {}, {}, {}
    for name, estimator in classifiers.items():
        model = Pipeline([("preprocessor", make_preprocessor(available_numeric, available_categorical)),
                          ("model", estimator)])
        model.fit(x_train, band_train)
        prediction = model.predict(x_test)
        accuracy = accuracy_score(band_test, prediction)
        weighted_f1 = f1_score(band_test, prediction, average="weighted")
        print(f"\n{name}: accuracy={accuracy:.4f}, weighted_F1={weighted_f1:.4f}")
        print(classification_report(band_test, prediction, zero_division=0))
        trained_classifiers[name] = model
        classifier_metrics[name] = {"accuracy": float(accuracy), "weighted_f1": float(weighted_f1)}
    for name, estimator in regressors.items():
        model = Pipeline([("preprocessor", make_preprocessor(available_numeric, available_categorical)),
                          ("model", estimator)])
        model.fit(x_train, price_train)
        report_regression(name, model, x_test, price_test)
        trained_regressors[name] = model

    payload = {
        "feature_columns": feature_columns,
        "numeric_columns": available_numeric,
        "classifiers": trained_classifiers,
        "regressors": trained_regressors,
        "classifier_metrics": classifier_metrics,
        "price_band_definition": {"budget": "<= $10,000", "mid-range": "$10,001-$25,000", "premium": "> $25,000"},
    }
    with open("models.pkl", "wb") as handle:
        pickle.dump(payload, handle)
    print("\nSaved trained pipelines to models.pkl")


if __name__ == "__main__":
    main()
