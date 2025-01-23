import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

MODEL_PATH = "../data/model.pkl"


def train_best_model(df: pd.DataFrame):
    X = df[["Temperature", "Run_Time"]]
    y = df['Downtime_Flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }
    best_model = None
    best_score = 0

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        if score > best_score:
            best_model = clf
            best_score = score

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    return {"message": "Model trained successfully", "accuracy": best_score}


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_downtime(model, data):
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)[0]
    probability = max(model.predict_proba(input_data)[0])
    return {"Downtime": "Yes" if prediction == 1 else "No", "Confidence": round(probability, 2)}
