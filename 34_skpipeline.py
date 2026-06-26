import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_csv("train.csv")

# select features 
features = ["Pclass", "Sex", "Age", "Fare"]
target = "Survived"
df = df[features + [target]]

# split data
x_train, x_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

numeric_features = ["Age", "Fare"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_features = ["Sex", "Pclass"]
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

pipe = Pipeline(steps=[
     ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)

# Save the complete pipeline
joblib.dump(pipe, "titanic_pipeline.pkl")

print("Pipeline saved successfully!")

print("Accuracy:", accuracy_score(y_test, y_pred))
