import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.componenets.data_transformation import DataTransformation
from src.componenets.data_transformation import DataTransformationConfig
from src.componenets.model_trainer import ModelTrainer
from src.componenets.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    directory: str = 'artifacts'

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Dataset read as pandas DataFrame")

            os.makedirs(self.ingestion_config.directory, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts/data.csv")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info("Train data saved to artifacts/train.csv")

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Test data saved to artifacts/test.csv")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e, sys) from e
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, raw_data = obj.initiate_data_ingestion()
    print(f"Train Data Path: {train_data}")
    print(f"Test Data Path: {test_data}")
    print(f"Raw Data Path: {raw_data}")

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)

    print(f"Train Array Shape: {train_arr.shape}")
    print(f"Test Array Shape: {test_arr.shape}")
    print(f"Preprocessor Path: {preprocessor_path}")

    model_trainer = ModelTrainer()
    best_model_name, model_report = model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)
    print(f"Best Model Name: {best_model_name}")
    print(f"Model Report: {model_report}")