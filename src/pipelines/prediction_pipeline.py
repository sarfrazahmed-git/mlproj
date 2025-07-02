import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import load_object

class PredictionPipeline:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = load_object(self.model_path)
        self.preprocessor = load_object(self.preprocessor_path)
    def predict(self, input_data):
        try:
            logging.info("Starting prediction process...")
            
            input_data_transformed = self.preprocessor.transform(input_data)
            logging.info("Input data transformed successfully.")
            
            predictions = self.model.predict(input_data_transformed)
            logging.info("Predictions made successfully.")
            
            return predictions        
        except Exception as e:
            raise CustomException(e, sys) from e

class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

