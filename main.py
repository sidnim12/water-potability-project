```python
# -----------------------------
# Enhanced Water Potability Project with Visualizations
# -----------------------------

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Check current working directory
try:
    print("Current working directory:", os.getcwd())
except Exception as e:
    print("Error checking working directory:", e)

# 2. Load dataset
try:
    df = pd.read_csv("water_potability.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'water_potability.csv' not found.")
    exit()
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# 3. Explore dataset
print("----- HEAD -----")
print(df.head())
print("----- INFO -----")
print(df.info())
print("----- DESCRIPTION -----")
print(df.describe())

# 4. Visualize missing values
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# 5. Handle missing values
df_clean = df.fillna(df.mean())
print("Missing values filled with column means.")

# 6. Target distribution plot
plt.figure(figsize=(6,4))
sns.countplot(x='Potability', data=df_clean)
plt.title("Potability Distribution")
plt.xlabel("Potability (0 = Not Potable, 1 = Potable)")
plt.ylabel("Count")
plt.show()

# 7. Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 8. Split dataset into features and target
X = df_clean.drop('Potability', axis=1)
y = df_clean['Potability']

# 9. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 10. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 11. Make predictions
y_pred = model.predict(X_test)

# 12. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("----- MODEL ACCURACY -----")
print(f"Accuracy: {accuracy:.2f}")

print("----- CONFUSION MATRIX -----")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("----- CLASSIFICATION REPORT -----")
print(classification_report(y_test, y_pred))

# 13. Confusion matrix heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

