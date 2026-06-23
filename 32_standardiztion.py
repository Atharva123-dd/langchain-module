# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import seaborn as sns

# # Load data
# df = pd.read_csv("train.csv")

# # Features and target
# X = df.drop("Survived", axis=1)
# y = df["Survived"]

# # Split data
# x_train, x_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Numeric columns
# num_cols = x_train.select_dtypes(include=["int64", "float64"]).columns

# # Scale
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(x_train[num_cols])

# x_train_scaled = pd.DataFrame(
#     scaled_data,
#     columns=num_cols
# )

# # Before Scaling
# plt.figure(figsize=(12, 5))

# # plt.subplot(1,2,1)
# # plt.scatter(x_train["Age"], x_train["Fare"], alpha=0.5)
# # plt.title("Before Scaling")
# # plt.xlabel("Age")
# # plt.ylabel("Fare")


# # # After Scaling
# # plt.subplot(1,2,2)
# # plt.scatter(x_train_scaled["Age"], x_train_scaled["Fare"], alpha=0.5)
# # plt.title("After Scaling")
# # plt.xlabel("Age (Scaled)")
# # plt.ylabel("Fare (Scaled)")
# sns.kdeplot(x_train["Age"].dropna(), label="Age Before", fill=True)

# sns.kdeplot(x_train_scaled["Age"].dropna(), label="Age After", fill=True)

# plt.title("Age Distribution Comparison (Before vs After Scaling)")
# plt.tight_layout()
# plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("train.csv")

# Drop useless text columns (important for Titanic)
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

# Convert categorical to numeric
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
# Features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

x_train = x_train.fillna(x_train.mean(numeric_only=True))
x_test = x_test.fillna(x_test.mean(numeric_only=True))
# Scale numeric columns
num_cols = x_train.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()

x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])

# -----------------------
# Train Logistic Regression
# -----------------------
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed report
print(classification_report(y_test, y_pred))