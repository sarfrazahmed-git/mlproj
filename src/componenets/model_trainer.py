from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import os
from xgboost import XGBRegressor
import sys
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models
from dataclasses import dataclass
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self, train_arr, test_arr,preprocess_path):
        try:
            logging.info("Splitting training and testing arrays into features and target variable")
            X_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            X_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'KNN': KNeighborsRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor()
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            best_model_name = max(model_report, key=lambda x: model_report[x]['r2_score'])
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with R2 score: {model_report[best_model_name]['r2_score']}")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            #save_object(preprocess_path, preprocess_path)

            return best_model_name, model_report[best_model_name]

        except Exception as e:
            raise CustomException(e, sys) from e
