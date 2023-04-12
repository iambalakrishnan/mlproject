import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exceptions.exception import CustomException
from src.logs.log import logging


def data_split(data, test_size=0.2, random_state=42):
    train_set, test_set = train_test_split(data,test_size=test_size, random_state=random_state)
    return (train_set, test_set)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            grid = GridSearchCV(model, para, cv=3)
            grid.fit(X_train, y_train)
            
            model.set_params(**grid.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report   

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)