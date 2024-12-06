import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Define input and output directories
input_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpatinput_aligned" 
output_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpatoutput"

input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
X_data = []
for file in input_files:
    df = pd.read_csv(os.path.join(input_dir, file))
    df = df.set_index("name")["value"]  # Set 'name' as index and keep 'value'
    X_data.append(df)
X = pd.DataFrame(X_data).reset_index(drop=True)
X = X.fillna(0)  # Replace NaN values with 0 (or use another strategy)

output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])
y_data = []
for file in output_files:
    df = pd.read_csv(os.path.join(output_dir, file))
    # Extract Peak Power
    peak_power = df[(df["Component"] == "Processor") & (df["Metric"] == "Peak Power")]["Value"].values[0]
    y_data.append(peak_power)
y = np.array(y_data)

# Scale Features (X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators = 100, random_state = 42)


#uncomment this to see cross validation results
# mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# cv_scores = cross_val_score(model, X, y, cv=5, scoring=mae_scorer)


# print("Cross-Validation MAE Scores (Negative):", cv_scores)
# print("Mean MAE (Positive):", -np.mean(cv_scores))  # Convert back to positive
# print("Standard Deviation of MAE:", np.std(cv_scores))
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

#Display model coefficients
print(X.head())
print(y)
# Step 6: Plot Predictions vs Actual Values
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
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