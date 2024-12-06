
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
import pickle
# Define a function for feature-based data augmentation using actual data ranges
def augment_features(X, y, num_copies=10, variation_percentage=0.1):
    """
    Augments data by introducing variations in features based on actual data ranges.

    Parameters:
        X (pd.DataFrame): Original feature data.
        y (np.array): Original target values.
        num_copies (int): Number of synthetic copies to generate for each sample.
        variation_percentage (float): Percentage of variation (e.g., 0.1 means ±10%).

    Returns:
        X_augmented (pd.DataFrame): Augmented feature data.
        y_augmented (np.array): Augmented target values.
    """
    # Calculate feature ranges based on min and max values in the dataset
    feature_ranges = {}
    for col in X.columns:
        min_val = X[col].min()
        max_val = X[col].max()
        range_val = max_val - min_val
        feature_ranges[col] = (min_val - range_val * variation_percentage, max_val + range_val * variation_percentage)

    X_augmented = []
    y_augmented = []

    for _ in range(num_copies):
        X_copy = X.copy()
        for feature in X.columns:
            low, high = feature_ranges[feature]
            # Add random variation within the specified range
            X_copy[feature] += np.random.uniform(low, high, size=X.shape[0]) - X[feature].mean()
        X_augmented.append(X_copy)
        
        # Add noise to target values (optional)
        y_noisy = y + np.random.normal(0, 0.05 * np.std(y), size=y.shape)
        y_augmented.append(y_noisy)

    # Combine original and augmented data
    X_augmented = pd.concat([X] + X_augmented).reset_index(drop=True)
    y_augmented = np.hstack([y] + y_augmented)

    return X_augmented, y_augmented

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def hyperparameter_tuning(X_train, y_train, X_test, y_test, pickle_file='best_hyperparameters.pkl'):
    """
    Perform hyperparameter tuning for RandomForestRegressor using GridSearchCV.
    
    Parameters:
    X_train: Training data features
    y_train: Training data target values
    X_test: Test data features
    y_test: Test data target values
    
    Returns:
    best_model: The model with the best hyperparameters after tuning
    """

    if os.path.exists(pickle_file):
        # Load hyperparameters from the pickle file
        with open(pickle_file, 'rb') as f:
            best_params = pickle.load(f)
        print("Loaded hyperparameters from pickle file.")
        best_model = RandomForestRegressor(**best_params, random_state=42)

    else:
    # Define the model
        model = RandomForestRegressor(random_state=42)

        # Set up the parameter grid
        # param_grid = {
        #     'n_estimators': [50, 100, 150, 200],
        #     'max_depth': [None, 10, 20, 30, 40],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 5],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'bootstrap': [True, False]
        # }
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.2, 0.5],
            'bootstrap': [True, False]
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Print best hyperparameters
        best_params = grid_search.best_params_
        with open(pickle_file, 'wb') as f:
            pickle.dump(best_params, f)
        print("Saved best hyperparameters to pickle file.")

        # Get the best model
        best_model = grid_search.best_estimator_

        # Evaluate the best model
    # y_pred = best_model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # print(f"Mean Squared Error: {mse}")
    # print(f"R-squared: {r2}")

    return best_model

    # # Example usage:
    # best_model = hyperparameter_tuning(X_train, y_train, X_test, y_test)
