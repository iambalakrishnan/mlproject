import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.utils.util import save_object,evaluate_model
from src.exceptions.exception import CustomException
from src.logs.log import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","saved_model","model.pkl")

class ModelTrainer:
    """_summary_
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def Initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoosing":XGBRegressor(),
                "Catboost":CatBoostRegressor(verbose=False),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),   
            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test,y_test=y_test,models=models)
            logging.info("model Report generated")

            best_model_score = max(sorted(model_report.values()))
            logging.info("model Scores are saved")

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info("All the given models are underperforming, score for the all the models are under 60% ")
                raise CustomException("No Best model found because score for the all the models are under 60% ")
            logging.info("Best model on both training and test dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
                        )
            
            predicted = best_model.predict(X_test)
            r2=r2_score(y_test,predicted)
            return r2

        except Exception as e:
            raise CustomException(e,sys)