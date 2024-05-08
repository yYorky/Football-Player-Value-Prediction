from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
import sys
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    STAGE_NAME = "Data Ingestion stage"
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion_obj = DataIngestion()
        df = data_ingestion_obj.initiate_data_ingestion()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys)
    

    # STAGE_NAME = "Data Transformation stage"
    # try:
    #     logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #     data_transformation_obj = DataTransformation()
    #     X,y,_ = data_transformation_obj.initiate_data_transformation(df)
    #     logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    # except Exception as e:
    #     raise CustomException(e, sys)
    

    # STAGE_NAME = "Model Trainer stage"
    # try:
    #     logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #     model_trainer_obj = ModelTrainer()
    #     model_trainer_obj.initiate_model_trainer(X,y)
    #     logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    # except Exception as e:
    #     raise CustomException(e, sys)