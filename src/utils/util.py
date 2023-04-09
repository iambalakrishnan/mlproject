import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.model_selection import train_test_split

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