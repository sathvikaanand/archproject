
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from helperfunctions import augment_features, hyperparameter_tuning

# Define input and output directories
input_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpat_device_input_aligned"
output_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpat_device_output"

input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
X_data = []
for file in input_files:
    df = pd.read_csv(os.path.join(input_dir, file))
    df = df.set_index("name")["value"]  # Set 'name' as index and keep 'value'
    X_data.append(df)
X = pd.DataFrame(X_data).reset_index(drop=True)
#X = pd.DataFrame(X_data).reset_index(drop=True)[["machine_bits", "withPHY", "clock_rate", "device_type", "interconnect_projection_type", "number_of_cores", "vertical_nodes", "horizontal_nodes", "physical_address_width", "local_predictor_entries", "virtual_address_width",  "number_of_L2s", "number_ranks", "clockrate", "has_global_link"]]

X = X.fillna(0)  # Replace NaN values with 0 (or use another strategy)
print(X)
output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])
y_data = []
for file in output_files:
    df = pd.read_csv(os.path.join(output_dir, file))
    # Extract Peak Power
    peak_power = df[(df["Component"] == "Processor") & (df["Metric"] == "Peak Power")]["Value"].values[0]
    y_data.append(peak_power)
y = np.array(y_data)


# Generate augmented data
num_copies = 0 # Create 5 synthetic copies per original data point
X_augmented, y_augmented = augment_features(X, y, num_copies=num_copies, variation_percentage=0.1)
X = X_augmented
y = y_augmented
# Print results
print(f"Original Data Points: {X.shape[0]}")
print(f"Augmented Data Points: {X_augmented.shape[0]}")




# scaler_y = StandardScaler()
# y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
plt.hist(y, bins=20, edgecolor='black')
plt.title("Target Variable Distribution")
plt.xlabel("Peak Power (W)")
plt.ylabel("Frequency")
plt.show()

# Save column names before scaling or PCA
original_feature_names = X.columns


# Scale Features (X)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# pca = PCA(n_components=0.99)  # Retain 95% of variance
# X = pca.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = hyperparameter_tuning(X_train, y_train, X_test, y_test)


# model = RandomForestRegressor(n_estimators = 100, random_state = 42)


#uncomment this to see cross validation results
# mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# cv_scores = cross_val_score(model, X, y, cv=5, scoring=mae_scorer)


# print("Cross-Validation MAE Scores (Negative):", cv_scores)
# print("Mean MAE (Positive):", -np.mean(cv_scores))  # Convert back to positive
# print("Standard Deviation of MAE:", np.std(cv_scores))
model.fit(X_train, y_train)
# # After training the model, retrieve feature importances
feature_importances = model.feature_importances_

# Create a Series with the original feature names and their importances
important_features = pd.Series(feature_importances, index=original_feature_names).sort_values(ascending=False)

print("Top 10 Important Features:")
print(important_features.head(10))
plt.hist(y, bins=20, edgecolor='black')
plt.title("Target Variable Distribution")
plt.xlabel("Peak Power (W)")
plt.ylabel("Frequency")
plt.show()


# Step 5: Evaluate the Model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
root_mse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {root_mse}")
print(f"R-squared: {r2}")


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Cross-Validation R² Scores: {scores}")
print(f"Mean R²: {np.mean(scores)}")

#Display model coefficients
# print(X.head())
print(y)
# Step 6: Plot Predictions vs Actual Values
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], color="red", linestyle="--")
plt.xlabel("Actual Peak Power (W)")
plt.ylabel("Predicted Peak Power (W)")
plt.title("Actual vs Predicted Peak Power")
plt.show()

print("Input Files:")
for file in input_files:
    print(file)

print("\nOutput Files:")
for file in output_files:
    print(file)