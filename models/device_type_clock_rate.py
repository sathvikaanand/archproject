import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from helperfunctions import augment_features, hyperparameter_tuning

# Define input and output directories
input_dir_1 = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpat_device_input_aligned"
output_dir_1 = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpat_device_output"

input_dir_2 = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpat_input_aligned"
output_dir_2 = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpat_output"

# Function to load data from a directory
def load_data(input_dir, output_dir):
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])

    X_data = []
    for file in input_files:
        df = pd.read_csv(os.path.join(input_dir, file))
        df = df.set_index("name")["value"]  # Set 'name' as index and keep 'value'
        X_data.append(df)
    X = pd.DataFrame(X_data).reset_index(drop=True).fillna(0)

    y_data = []
    for file in output_files:
        df = pd.read_csv(os.path.join(output_dir, file))
        # Extract Peak Power
        peak_power = df[(df["Component"] == "Processor") & (df["Metric"] == "Peak Power")]["Value"].values[0]
        y_data.append(peak_power)
    y = np.array(y_data)

    return X, y

# Load data from both directories
X1, y1 = load_data(input_dir_1, output_dir_1)
X2, y2 = load_data(input_dir_2, output_dir_2)

# Combine the datasets
X_combined = pd.concat([X1, X2], ignore_index=True).fillna(0)
y_combined = np.concatenate([y1, y2], axis=0)

# Generate augmented data
num_copies = 0  # Adjust as needed
X_augmented, y_augmented = augment_features(X_combined, y_combined, num_copies=num_copies, variation_percentage=0.1)
X = X_augmented
y = y_augmented

# Print dataset info
print(f"Original Data Points: {X1.shape[0] + X2.shape[0]}")
print(f"Augmented Data Points: {X.shape[0]}")

# Plot target variable distribution
plt.hist(y, bins=20, edgecolor='black')
plt.title("Target Variable Distribution")
plt.xlabel("Peak Power (W)")
plt.ylabel("Frequency")
plt.show()

# Save column names before scaling
original_feature_names = X.columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model using hyperparameter tuning
model = hyperparameter_tuning(X_train, y_train, X_test, y_test)

# Fit the model
model.fit(X_train, y_train)

# Retrieve feature importances
feature_importances = model.feature_importances_
important_features = pd.Series(feature_importances, index=original_feature_names).sort_values(ascending=False)

# Display top 10 features
print("Top 10 Important Features:")
print(important_features.head(10))

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
root_mse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {root_mse}")
print(f"R-squared: {r2}")

# Plot predictions vs actual values
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual Peak Power (W)")
plt.ylabel("Predicted Peak Power (W)")
plt.title("Actual vs Predicted Peak Power")
plt.show()

# Cross-validation scores
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Cross-Validation R² Scores: {scores}")
print(f"Mean R²: {np.mean(scores)}")

