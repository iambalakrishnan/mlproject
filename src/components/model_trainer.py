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
                "AdaBoost Regressor":AdaBoostRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose=False),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),   
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boost":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Linear Regression":{},
                "K-Neighbors":{}

                
            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test,y_test=y_test,models=models, param=params)
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