
import pickle
import time
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

MODEL_PATH = "dry_bean_nonlinear_svm_model.pkl"


def main():
    start_time = time.time()

    # 1. Download the Dry Bean dataset directly from UCI
    print("Downloading Dry Bean dataset from UCI...")

    dataset = fetch_ucirepo(id=602)

    X = dataset.data.features
    y = dataset.data.targets.iloc[:, 0]

    # Ensure features are numerical and data is valid
    X = X.apply(pd.to_numeric, errors="coerce")
    valid_rows = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    print("Dataset downloaded successfully!")
    print("Dataset shape:", X.shape)
    print("Features:", list(X.columns))
    print("Classes:", sorted(y.unique()))

    # 2. Encode the seven bean varieties as integer labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 4. Build the Non-Linear SVM pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            cache_size=1000,
        )),
    ])

    # 5. Train the model
    print("\nTraining Non-Linear SVM (RBF kernel)...")
    model.fit(X_train, y_train)

    # 6. Evaluate the model
    print("\nEvaluating model...")
    predictions = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=list(range(len(encoder.classes_))),
        target_names=encoder.classes_,
        zero_division=0,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(encoder.classes_))),
    ))

    # 7. Save the model and class encoder together
    artifact = {
        "model": model,
        "label_encoder": encoder,
        "feature_names": list(X.columns),
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(artifact, file)

    print(f"\nModel saved successfully: {MODEL_PATH}")

    # 8. Verify that the saved model can make predictions
    with open(MODEL_PATH, "rb") as file:
        loaded_artifact = pickle.load(file)

    loaded_model = loaded_artifact["model"]
    loaded_encoder = loaded_artifact["label_encoder"]

    sample = X_test.iloc[[0]]
    predicted_class = loaded_model.predict(sample)[0]
    predicted_name = loaded_encoder.inverse_transform(
        [predicted_class]
    )[0]

    actual_name = loaded_encoder.inverse_transform(
        [y_test[0]]
    )[0]

    print("\nSaved model verification:")
    print("Actual bean variety:", actual_name)
    print("Predicted bean variety:", predicted_name)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
