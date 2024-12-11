import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb

input_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpat_input_aligned" # Replace with your input directory path
output_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpat_output"  # Replace with your output directory path

input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
X_data = []
for file in input_files:
    df = pd.read_csv(os.path.join(input_dir, file))
    df = df.set_index("name")["value"]  # Set 'name' as index and keep 'value'
    X_data.append(df)

#X = pd.DataFrame(X_data).reset_index(drop=True)
X = pd.DataFrame(X_data).reset_index(drop=True)[["withPHY", "target_core_clockrate", "mc_clock", "output_ports", "clock_rate", "MUL_per_core", "input_ports", "number_hardware_threads", "instruction_window_scheme"]]

X = X.fillna(0)  # Replace NaN values with 0 (or use another strategy)


output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])
y_data = []
for file in output_files:
    df = pd.read_csv(os.path.join(output_dir, file))
    # Extract Peak Power
    peak_power = df[(df["Component"] == "Processor") & (df["Metric"] == "Peak Power")]["Value"].values[0]
    y_data.append(peak_power)
y = np.array(y_data)



# Step 3: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train XGBoost Model
xgboost_model = xgb.XGBRegressor(
    n_estimators=100,        # Number of trees
    max_depth=6,             # Maximum depth of each tree
    learning_rate=0.1,       # Step size shrinkage
    subsample=0.8,           # Subsample ratio of training instance
    colsample_bytree=0.8,    # Subsample ratio of columns when constructing trees
    random_state=42
)



# Cross-validation
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
cv_scores = cross_val_score(xgboost_model, X, y, cv=5, scoring=mae_scorer)

print("Cross-Validation MAE Scores (Negative):", cv_scores)
print("Mean MAE (Positive):", -np.mean(cv_scores))  # Convert back to positive
print("Standard Deviation of MAE:", np.std(cv_scores))


# Fit the model
xgboost_model.fit(X_train, y_train)
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgboost_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Display Top 10 Features
print("Top 10 Important Features (XGBoost):")
print(importance_df.head(10))

# Step 5: Evaluate the Model
y_pred = xgboost_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 6: Plot Predictions vs Actual Values
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual Peak Power (W)")
plt.ylabel("Predicted Peak Power (W)")
plt.title("Actual vs Predicted Peak Power")
plt.show()

# Debugging Outputs (Optional)
print("Input Files:")
for file in input_files:
    print(file)

print("\nOutput Files:")
for file in output_files:
    print(file)

