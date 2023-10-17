import os
import sys
from src.Project1.exception import CustonException
from src.Project1.logger import logging
import pandas as pd
from src.Project1.utils import read_sql_data
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts' , 'train.csv')
    test_data_path:str = os.path.join('artifacts' , 'test.csv')
    raw_data_path:str = os.path.join('artifacts' , 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            
            raw_df = read_sql_data()

            logging.info("Reading from MySQL database")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path) , exist_ok=True)

            raw_df.to_csv(self.ingestion_config.raw_data_path,index=False , header=True)

            train_set , test_set = train_test_split(raw_df , test_size=0.25 , random_state=19)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False , header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False , header=True)

            logging.info("Data Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustonException(e,sys)