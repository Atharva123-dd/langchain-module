import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================
# 1. Load Dataset
# =========================
df = pd.read_excel("student_data.xlsx")

print("Dataset Preview:")
print(df.head())


# =========================
# 2. Convert Target FIRST
# =========================
df["Result"] = df["Result"].map({
    "Fail": 0,
    "Pass": 1
})


# Check if both classes exist
print("\nClass distribution:")
print(df["Result"].value_counts())


# =========================
# 3. Define Features & Target
# =========================
X = df[
    [
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Score"
    ]
]

y = df["Result"]


# =========================
# 4. Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 5. Train Model
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# =========================
# 6. Evaluate Model
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")


# =========================
# 7. Make Prediction
# =========================
new_student = pd.DataFrame(
    [[3, 50, 8, 25]],
    columns=[
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Score"
    ]
)

prediction = model.predict(new_student)

print("\nPrediction:")

if prediction[0] == 1:
    print("PASS")
else:
    print("FAIL")


# =========================
# 8. Save Model
# =========================
joblib.dump(model, "student_result_model.pkl")

print("\nModel saved successfully!")