import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import numpy as np

df = pd.read_csv("data/processed/features.csv")
df = df.dropna(subset=["label"])
print(df["label"].value_counts())
print(df.groupby("label")["room_id"].nunique())

FEATURES = ["rt60", "c80", "d50", "ts"]
X = df[FEATURES].values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "SVM":          SVC(kernel="rbf", class_weight="balanced", random_state=42),
    "kNN":          KNeighborsClassifier(n_neighbors=5),
    "XGBoost":      XGBClassifier(random_state=42, verbosity=0),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    all_true, all_pred = [], []
    for train_idx, test_idx in skf.split(X, y_enc):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test  = scaler.transform(X[test_idx])
        model.fit(X_train, y_enc[train_idx])
        preds = model.predict(X_test)
        all_true.extend(y_enc[test_idx])
        all_pred.extend(preds)
    print(f"\n=== {model_name} ===")
    print(classification_report(all_true, all_pred, target_names=le.classes_))
