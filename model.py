from src.Project1.logger import logging 
from src.Project1.exception import CustonException
from src.Project1.components.data_ingestion import DataIngestion , DataIngestionConfig
from src.Project1.components.data_transformation import DataTransformation , DataTransformationConfig
from src.Project1.components.model_trainer import modeltrainerconfig , modeltariner
import sys 
import os 

if __name__ == "__main__":
    logging.info("The execution has started")
    
    try:
    
        data_ingestion = DataIngestion()
        train_data_path , test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr , test_arr , preprocessor_file_path = data_transformation.initiate_data_transformation(train_data_path , test_data_path)

        data_trainig = modeltariner()
        r2 = data_trainig.initiate_model_trainer(train_arr , test_arr)

        print(r2)

        
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustonException(e ,sys)