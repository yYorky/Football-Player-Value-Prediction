import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error
from math import sqrt

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X, y, models, cv=10):
    try:
        report = {}

        skf = StratifiedKFold(n_splits=cv, shuffle=True)

        for model_name, model in models.items():
            
            # Perform cross-validation
            rmse_scores = []
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Fit the model
                model.fit(X_train, y_train)

                # Predictions
                y_test_pred = model.predict(X_test)

                # Calculate RMSE
                rmse = sqrt(mean_squared_error(y_test, y_test_pred))
                rmse_scores.append(rmse)

            # Average RMSE across folds
            mean_rmse = sum(rmse_scores) / len(rmse_scores)

            report[model_name] = {
                'mean_rmse': mean_rmse,
                'rmse_scores': rmse_scores
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)