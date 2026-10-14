import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Download dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

columns = ["variance", "skewness", "curtosis", "entropy", "class"]
data = pd.read_csv(url, header=None, names=columns)

print("Dataset downloaded successfully!")
print("Dataset shape:", data.shape)

# 2. Separate features and labels
X = data[["variance", "skewness", "curtosis", "entropy"]]
y = data["class"].astype(int)

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Create and train Linear SVM model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", C=1.0))
])

print("Training Linear SVM model...")
model.fit(X_train, y_train)

# 5. Evaluate model
predictions = model.predict(X_test)

print("\nModel Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Genuine", "Counterfeit"]
))

# 6. Save trained model as PKL
with open("banknote_svm_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nSUCCESS! Model saved as banknote_svm_model.pkl")

# 7. Test loading the saved model
with open("banknote_svm_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

sample = [[3.6216, 8.6661, -2.8073, -0.44699]]
result = loaded_model.predict(sample)[0]

print("Sample prediction:", "Genuine" if result == 0 else "Counterfeit")
