import os
import sys
from src.Project1.exception import CustonException
from src.Project1.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pymysql
import numpy as np
import sqlalchemy as sa
import warnings

warnings.filterwarnings('ignore')

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading Started")
    try:
        mydb = pymysql.connect(
            host = host,
            user = user,    
            password=password,
            db = db
        )

        logging.info("Connection Successful")
        df = pd.read_sql_query("select * from students" , mydb)
        # print(df.head())

        return df


    except Exception as e:
        raise CustonException(e ,sys)
    

import pickle

def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path , exist_ok=True)

        with open(file_path , "wb") as file_obj:
            pickle.dump(obj , file_obj)
    except Exception as e:
        raise CustonException(e ,sys)
    

def load_object(file_path):
    try:
        with open(file_path , 'rb') as file_obj:
            return pickle.load(file_obj)



    except Exception as e:
        raise CustonException(e,sys)









    
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error

def evaluate_model(test , pred):
    mae = mean_absolute_error(test,pred)
    mse = mean_squared_error(test,pred)
    rmse = np.sqrt(mean_squared_error(test,pred))
    r2 = r2_score(test,pred)
    return mae , rmse , r2

def evaluate_models(x_train , y_train , x_test , y_test , models , params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(params.keys())[i]]

            gs = GridSearchCV(model ,para , cv=3)
            gs.fit(x_train , y_train)

            model.set_params(**gs.best_params_)

            model.fit(x_train , y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            training_score = r2_score(y_train , y_train_pred)
            testing_score = r2_score(y_test , y_test_pred)

            report[list(models.keys())[i]] = testing_score

        return report
    
    except Exception as e :
        raise CustonException(e,sys)