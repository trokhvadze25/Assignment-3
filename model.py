# Darknet Traffic Analysis and Activity Prediction
# Student-style ML project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("Darknet.csv")

print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# ----------------------------
# 2. Exploratory Data Analysis
# ----------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df['Label1'].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(x='Label1', data=df)
plt.title("Activity Label Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------
# 3. Data Preprocessing
# ----------------------------
le = LabelEncoder()
df['Label1'] = le.fit_transform(df['Label1'])

X = df.drop('Label', axis=1)
y = df['Label1']

# ----------------------------
# 4. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# 5. Model Training
# ----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ----------------------------
# 6. Model Evaluation
# ----------------------------
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Feature Importance
# ----------------------------
importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()

# ----------------------------
# 8. Sample Prediction
# ----------------------------
sample = X_test.iloc[0:1]
prediction = model.predict(sample)

print("\nPredicted Activity:", le.inverse_transform(prediction))
