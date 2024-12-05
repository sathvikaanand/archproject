import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt
import os

# Define input and output directories
inputs_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpatinputfiltered" 
outputs_dir = "/Users/keshavt/Documents/CS254_Final_Proj/archproject/outputs/mcpatoutput"

# Load input and output data
X = []
y = []

for input_file in os.listdir(inputs_dir):
    if input_file.endswith(".csv"):
        # Load input data
        input_path = os.path.join(inputs_dir, input_file)
        datainputs = pd.read_csv(input_path).iloc[:, -2:]
        X.append(datainputs['value'].values)

        output_file = input_file.replace("xml", "3")
        output_path = os.path.join(outputs_dir, output_file)
        # Corresponding output file
        if os.path.exists(output_path):
            dataoutputs = pd.read_csv(output_path).iloc[1, -1:]
            y.append(dataoutputs.values)

# Convert to DataFrame and clean
X = pd.DataFrame(X).dropna(axis='columns')
y = pd.DataFrame(y).dropna(axis='columns')
y = np.ravel(y)

print(f"Features (X):\n{X}")
print(f"Target (y):\n{y}")
print(f"Number of Samples: {len(X)}, Target Samples: {len(y)}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

# Evaluate the model
mse = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Random Forest Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mse}")
print(f"R-squared (R²): {r2}")


# Print the tree structure
tree_rules = export_text(dt_model, feature_names=X.columns.to_list())
print(tree_rules)

# Visualize the tree (graphical)
plt.figure(figsize=(12, 8))
plot_tree(dt_model, feature_names=X.columns, filled=True, fontsize=10)
plt.show()

# Visualize Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Target')
plt.ylabel('Predicted Target')
plt.title('Actual vs Predicted (Random Forest)')
plt.show()

