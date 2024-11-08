import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# RANDOM FAKE DATA

np.random.seed(0)
num_samples = 100
frequency = np.random.uniform(1.5, 3.5, num_samples)  # Random frequencies between 1.5 and 3.5 GHz
voltage = np.random.uniform(0.8, 1.3, num_samples)    # Random voltages between 0.8 and 1.3 V
cache_size = np.random.randint(512, 8192, num_samples)  # Random cache sizes between 512KB and 8192KB

# FAKE DATA 
energy_consumption = (0.5 * frequency + 2.5 * voltage + 0.001 * cache_size
                      + np.random.normal(0, 0.5, num_samples))

data = pd.DataFrame({
    'Frequency (GHz)': frequency,
    'Voltage (V)': voltage,
    'Cache Size (KB)': cache_size,
    'Energy Consumption (W)': energy_consumption
})

X = data[['Frequency (GHz)', 'Voltage (V)', 'Cache Size (KB)']]
y = data['Energy Consumption (W)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Energy Consumption (W)')
plt.ylabel('Predicted Energy Consumption (W)')
plt.title('Actual vs Predicted Energy Consumption')
plt.show()
