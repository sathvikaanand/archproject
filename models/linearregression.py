import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Define input and output directories
inputs_dir = "/Users/sathv/archproject/outputs/mcpatinput"
outputs_dir = "/Users/sathv/archproject/outputs/mcpatoutput"

X = []
y = []
# Iterate through all input and output files
for input_file in os.listdir(inputs_dir):
    if input_file.endswith(".csv"):
        # Load input data
        input_path = os.path.join(inputs_dir, input_file)
        datainputs = pd.read_csv(input_path).iloc[:, -2:]
        X.append(datainputs['value'].values)

        output_file = input_file.replace("xml", "3")
        output_path = os.path.join(outputs_dir, output_file)
        # Corresponding output file
        print(output_path)

        if os.path.exists(output_path):
            print(output_path)
            dataoutputs = pd.read_csv(output_path).iloc[1, -1:]
            y.append(dataoutputs.values)
# print(len(y))
# # Save the combined DataFrame to a CSV file
# output_csv_path = "/Users/sathv/archproject/mcpat/combined_data.csv"
# combined_df.to_csv(output_csv_path, index=False)

# print(f"Combined DataFrame saved to {output_csv_path}")
# datainputs = pd.read_csv("/Users/sathv/archproject/mcpat/parseXML/xeon_inputs.csv").iloc[:,-2:]
# dataoutputs = pd.read_csv("/Users/sathv/archproject/mcpat_data.csv").iloc[:,-1:]

X = pd.DataFrame(X)
X = X.dropna(axis='columns')
y = pd.DataFrame(y)
y = y.dropna(axis='columns')
y = np.ravel(y)
print(X)
print(y)
print(len(X), len(y))

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# LR = LogisticRegression()
# KNN = KNeighborsClassifier()
# NB = GaussianNB()
# LSVM = LinearSVC()
model = MLPRegressor()
# DT = DecisionTreeClassifier()
# RF = RandomForestClassifier()

# LR_fit = LR.fit(X_train, Y_train)
# KNN_fit = KNN.fit(X_train, Y_train)
# NB_fit = NB.fit(X_train, Y_train)
# LSVM_fit = LSVM.fit(X_train, Y_train)
fit = model.fit(X_train, Y_train)
# DT_fit = DT.fit(X_train, Y_train)
# RF_fit = RF.fit(X_train, Y_train)

# LR_pred = LR_fit.predict(X_test)
# KNN_pred = KNN_fit.predict(X_test)
# NB_pred = NB_fit.predict(X_test)
# LSVM_pred = LSVM_fit.predict(X_test)
y_pred = fit.predict(X_test)
# DT_pred = DT_fit.predict(X_test)
# RF_pred = RF_fit.predict(X_test)

# from sklearn.metrics import accuracy_score
# print("Logistic Regression is %f percent accurate" % (accuracy_score(LR_pred, Y_test)*100))
# print("KNN is %f percent accurate" % (accuracy_score(KNN_pred, Y_test)*100))
# print("Naive Bayes is %f percent accurate" % (accuracy_score(NB_pred, Y_test)*100))
# print("Linear SVMs is %f percent accurate" % (accuracy_score(LSVM_pred, Y_test)*100))
# print("Non Linear SVMs is %f percent accurate" % (accuracy_score(NLSVM_pred, Y_test)*100))
# print("Decision Trees is %f percent accurate" % (accuracy_score(DT_pred, Y_test)*100))
# print("Random Forests is %f percent accurate" % (accuracy_score(RF_pred, Y_test)*100))


# model = sm.OLS(y_train, X_train).fit()

# # # Print the summary
# print(model.summary())


# print("RANDOM FOREST: ")
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

mse = mean_absolute_error(Y_test,  y_pred)
r2 = r2_score(Y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)



# # print("LINEAR REGRESSION: ")
# # model = LinearRegression()
# # model.fit(X_train, y_train)
# # y_pred = model.predict(X_test)
# print(y_pred)


# mse = mean_absolute_error(y_test,  y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Mean Squared Error:", mse)
# print("R-squared:", r2)
# # print("Coefficients:", model.coef_)
# # print("Intercept:", model.intercept_)

plt.scatter(Y_test, y_pred, color='blue')
plt.plot([Y_test.min(), Y_test.max()], [y_pred.min(), y_pred.max()], 'r--', lw=2)
plt.xlabel('Actual Energy Consumption (W)')
plt.ylabel('Predicted Energy Consumption (W)')
plt.title('Actual vs Predicted Energy Consumption')
plt.show()
