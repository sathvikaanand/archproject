import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Define input and output directories
input_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpat_input_aligned"
output_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpat_output"

# Step 1: Load and Prepare the Data
input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
X_data = []
for file in input_files:
    df = pd.read_csv(os.path.join(input_dir, file))
    df = df.set_index("name")["value"]
    X_data.append(df)
X = pd.DataFrame(X_data).reset_index(drop=True).fillna(0)

output_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])
y_data = []
for file in output_files:
    df = pd.read_csv(os.path.join(output_dir, file))
    peak_power = df[(df["Component"] == "Processor") & (df["Metric"] == "Peak Power")]["Value"].values[0]
    y_data.append(peak_power)
y = np.array(y_data)



# Save column names before scaling
original_feature_names = X.columns

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train a Random Forest Model to Get Feature Importances
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Get Feature Importances and Select Top 10 Features
feature_importances = rf_model.feature_importances_
important_features = pd.Series(feature_importances, index=original_feature_names).sort_values(ascending=False)
top_10_features = important_features.head(5).index
print("Top 10 Features:", top_10_features)

# Filter Data for Top 10 Features
X_train_top10 = pd.DataFrame(X_train, columns=original_feature_names)[top_10_features]
X_test_top10 = pd.DataFrame(X_test, columns=original_feature_names)[top_10_features]

# Step 6: Train a Decision Tree Regressor on Top 10 Features
dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_model.fit(X_train_top10, y_train)

# Step 7: Evaluate the Decision Tree Model
y_pred = dt_model.predict(X_test_top10)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (Decision Tree): {mse}")
print(f"R-squared (Decision Tree): {r2}")

# Step 8: Visualize the Decision Tree
plt.figure(figsize=(15, 8))
plot_tree(
    dt_model,
    feature_names=top_10_features,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization")
plt.show()

# Step 9: Text Representation of the Decision Tree
tree_rules = export_text(dt_model, feature_names=list(top_10_features))
print(tree_rules)

# Step 10: Plot Predictions vs Actual Values
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual Peak Power (Scaled)")
plt.ylabel("Predicted Peak Power (Scaled)")
plt.title("Actual vs Predicted Peak Power (Decision Tree)")
plt.show()
