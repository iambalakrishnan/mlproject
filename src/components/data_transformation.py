import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exceptions.exception import CustomException
from src.logs.log import logging
from src.utils.util import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','data_preprocessor','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """This Function is responsible for data transformation 

        Raises:
            CustomException: Custom error

        Returns:
            data: preprocessed data
        """
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                                    'gender',
                                    'race_ethnicity',
                                    'parental_level_of_education',
                                    'lunch',
                                    'test_preparation_course'
                                    ]
            

            numerical_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ])
            logging.info('Numerical column Scaleing completed')

            categorical_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
                ("scaler",StandardScaler(with_mean=False))
                ])
            logging.info('Categorical column encoding completed')

            logging.info(f'Categorical columns: {categorical_columns}')
            logging.info(f'Numerical columns: {numerical_columns}')


            preprocessor = ColumnTransformer(
                [
                ("numerical_pipeline", numerical_pipeline, numerical_columns),
                ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def Initiate_data_transformation(self, train_path, test_path):
        """_summary_

        Args:
            train_path (_type_): _description_
            test_path (_type_): _description_
        """
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info(f"Read the traing data from : {train_path}")
            logging.info(f"Read the test data from : {test_path}")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['writing_score','reading_score']

            input_feature_train_data = train_data.drop(columns=[target_column_name], axis=1)
            target_feature_train_data = train_data[target_column_name]

            input_feature_test_data = test_data.drop(columns=[target_column_name], axis=1)
            target_feature_test_data = test_data[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_array=preprocessor_obj.fit_transform(input_feature_train_data)
            input_feature_test_array=preprocessor_obj.transform(input_feature_test_data)

            train_array = np.c_[input_feature_train_array, np.array(target_feature_train_data)]
            test_array = np.c_[input_feature_test_array, np.array(target_feature_test_data)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessor_obj
            )

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_ob_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)