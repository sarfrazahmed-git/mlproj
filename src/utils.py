import os
import sys

import pandas as pd
import numpy as np
import dill
from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException

def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e

def evaluate_models(X_train, y_train, X_test, y_test, models):
    model_report = {}
    for model_name, model in models.items():
        logging.info(f"Training {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        model_report[model_name] = {
            'r2_score': r2,
            'mean_absolute_error': mae,
            'mean_squared_error': mse,
            'root_mean_squared_error': rmse
        }
        logging.info(f"{model_name} - R2: {r2}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
    return model_report