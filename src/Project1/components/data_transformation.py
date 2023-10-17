import sys
import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.Project1.utils import save_object
from src.Project1.exception import CustonException
from src.Project1.logger import logging
import os
from dataclasses import dataclass 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts' , 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            numerical_columns = ['reading_score', 'writing_score']

            num_pipeline = Pipeline(steps=[
                ("imputer" , SimpleImputer(strategy='median')),
                ('scaler' , StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer" , SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder" , OneHotEncoder()),
                ('scaler' , StandardScaler(with_mean=False))
            ])

            logging.info(f'Categorical columns:{categorical_columns}')
            logging.info(f'Numerical columns:{numerical_columns}')

            preprocesor = ColumnTransformer(
                [
                    ('numerical_pipeline' , num_pipeline , numerical_columns),
                    ('categorical_pipeline' , cat_pipeline , categorical_columns)

                ]
            )

            return preprocesor

        except Exception as e :
            raise CustonException(e,sys)
        
    def initiate_data_transformation(self , train_path , test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("reading training and testing file")

            preprocessing_obj = self.get_data_transformer_object()

            target = "math_score"

            # dividing training dataset

            input_features_train_df = train_df.drop(columns = [target] , axis =1)
            target_feature_train_df = train_df[target]

            #dividing testing datset

            input_features_test_df = test_df.drop(columns = [target] , axis =1)
            target_feature_test_df = test_df[target]

            logging.info('applying preprocessing on data')

            input_train_data = preprocessing_obj.fit_transform(input_features_train_df)
            input_test_data = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_train_data , np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_test_data , np.array(target_feature_test_df)
            ]

            logging.info(f'saved preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            logging.info(f'Data transformation complete')

            return (train_arr , test_arr , self.data_transformation_config.preprocessor_obj_file_path) 




        except Exception as e :
            raise CustonException(e, sys)