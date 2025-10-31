import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Set the filename for the uploaded dataset
DATASET_FILENAME = "diabetes_young_adults_india.csv"
TARGET_COLUMN = 'Diabetes_Type' # The column we are trying to predict

# Check if the dataset file exists in the current environment
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

# Display initial information
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nData Info:")
df.info()

# --- 2. Data Cleaning and Feature Engineering ---

# Drop irrelevant columns (assuming 'ID' is just an index)
if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)
    print("\n'ID' column dropped.")

# Handle missing values: Simple imputation (e.g., forward fill)
print("\nMissing Values before imputation:")
print(df.isnull().sum().sum())
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True) # Catch any remaining NaN at the start
print(f"Missing Values after imputation: {df.isnull().sum().sum()}")


# Clean up the 'Diabetes_Type' column: Treat 'None' as 'No Diabetes'
df[TARGET_COLUMN] = df[TARGET_COLUMN].replace('None', 'No Diabetes')
print(f"\nTarget Value Counts:\n{df[TARGET_COLUMN].value_counts()}")

# --- 3. Encoding Categorical Variables ---

# Separate features (X) and target (y)
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Label Encoding for the Target Variable (Classification)
# 0: No Diabetes, 1: Type 1, 2: Type 2
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"\nTarget classes mapped to: {list(label_encoder.classes_)} -> {label_encoder.transform(label_encoder.classes_)}")

# One-Hot Encoding for remaining Categorical Features
X = pd.get_dummies(X, drop_first=True)
print(f"\nFeature set shape after One-Hot Encoding: {X.shape}")

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

# --- 5. Build the Neural Network Model (Keras) ---
input_shape = X_train.shape[1]

# Sequential Model: Simple Deep Learning architecture
model = Sequential([
    # Input layer and first hidden layer
    Dense(128, activation='relu', input_shape=(input_shape,)),
    Dropout(0.3), # Dropout for regularization

    # Second hidden layer
    Dense(64, activation='relu'),
    Dropout(0.3),

    # Third hidden layer
    Dense(32, activation='relu'),

    # Output layer: uses softmax for multi-class classification
    Dense(num_classes, activation='softmax')
])

# Compile the model
# Using Adam optimizer and sparse_categorical_crossentropy because y is integer encoded
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print("Model built and compiled.")

# --- 6. Train the Model ---
# Use early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print("\n--- 6. Training the Neural Network ---")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1, # 10% of training data used for validation
    callbacks=[early_stopping],
    verbose=0 # Run silently to keep output clean, set to 1 for progress bar
)

print("Training finished. Best weights restored.")

# --- 7. Model Evaluation ---
print("\n--- 7. Model Evaluation ---")

# Evaluate on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Generate predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Inverse transform predictions for readability
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))

# --- 8. Visualization of Training History ---

# Plotting loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# --- Conclusion ---
print("\n=======================================")
print("  ✅ NEURAL NETWORK TRAINING COMPLETE  ")
print("=======================================")
print(f"Final Model Accuracy on Test Set: {accuracy*100:.2f}%")
print("Review the plots for training stability (Loss & Accuracy).")
print("You now have a trained model ready for deployment or further hyperparameter tuning.")
