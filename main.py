import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
DATASET_FILENAME = "diabetes_young_adults_india.csv"
TARGET_COLUMN = 'Diabetes_Type' # The column we are trying to predict

# Check for file existence
if not os.path.exists(DATASET_FILENAME):
    print(f"ERROR: The file '{DATASET_FILENAME}' was not found.")
    print("Please make sure the dataset is accessible in the same directory as this script.")
    exit()

# --- 1. Data Loading and Initial Inspection ---
print("--- 1. Data Loading and Initial Inspection ---")
try:
    df = pd.read_csv(DATASET_FILENAME)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Drop irrelevant columns (assuming 'ID' is just an index)
if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)
    print("\n'ID' column dropped.")

# --- 2. Data Cleaning and Preprocessing ---

# Handle missing values: Forward and Backward Fill
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
print(f"\nMissing Values after imputation: {df.isnull().sum().sum()}")

# Clean up the 'Diabetes_Type' column: Treat 'None' as 'No Diabetes'
df[TARGET_COLUMN] = df[TARGET_COLUMN].replace('None', 'No Diabetes')
print(f"\nTarget Value Counts:\n{df[TARGET_COLUMN].value_counts()}")

# --- 3. Encoding Categorical Variables ---

# Separate features (X) and target (y)
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Label Encoding for the Target Variable (Classification)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
target_classes = list(label_encoder.classes_)
print(f"\nTarget classes mapped to: {target_classes}")

# One-Hot Encoding for remaining Categorical Features
X = pd.get_dummies(X, drop_first=True)
print(f"Feature set shape after One-Hot Encoding: {X.shape}")

# --- 4. Feature Scaling and Train-Test Split ---

# Scale Numerical Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# --- 5. Build and Train Random Forest Model ---
print("\n--- 5. Training Random Forest Model ---")

# Initialize and train the Random Forest Classifier
# n_estimators=200 for better performance, max_depth=10 to control complexity
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced' # Helps with class imbalance
)
rf_model.fit(X_train, y_train)

print("Random Forest Model trained successfully.")

# --- 6. Model Evaluation ---
print("\n--- 6. Model Evaluation ---")

# Generate predictions
y_pred = rf_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_classes))

# Confusion Matrix Visualization 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_classes,
    yticklabels=target_classes
)
plt.title("Confusion Matrix (Random Forest)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# --- 7. Feature Importance ---
print("\n--- 7. Feature Importance ---")
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = feature_importances.sort_values(ascending=False).head(10)

print("Top 10 Most Important Features:")
print(top_features)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# --- Conclusion ---
print("\n=======================================")
print("  ✅ RANDOM FOREST CLASSIFIER COMPLETE  ")
print("=======================================")
print(f"Final Model Accuracy on Test Set: {accuracy*100:.2f}%")
print("The model has been successfully trained and evaluated without using TensorFlow.")
