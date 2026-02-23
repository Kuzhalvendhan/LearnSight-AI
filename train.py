import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("data/digital_learning_analytics_100k.csv")

# 🎯 Target
target = "mastery_score"

# Date processing
date_cols = ["enrollment_date", "last_activity_date"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df[col + "_year"] = df[col].dt.year
    df[col + "_month"] = df[col].dt.month
    df[col + "_day"] = df[col].dt.day

df.drop(columns=date_cols, inplace=True)

# Drop ID column (no predictive value)
df.drop(columns=["learner_id"], inplace=True)

# Label Encoding for categorical features
label_encoders = {}
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Split
X = df.drop(columns=[target])
y = df[target]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=75,       # reduce trees
    max_depth=10,          # strong depth limit
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",   # very important for size reduction
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

print(f"\nR² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# Save models
joblib.dump(model, "models/mastery_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("\nModel & preprocessors saved successfully.")
