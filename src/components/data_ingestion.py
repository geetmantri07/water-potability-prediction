import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.connections.s3_connection import s3_operations 
#from src.data_access.proj1_data import Proj1Data

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
        

    

    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            s3 = s3_operations()
            logging.info("Fetching data from s3")

            df = s3.fetch_file_from_s3("water_potability.csv")

            logging.info("Exporting data from s3")
            logging.info(f"Shape of dataframe: {df.shape}")

           # directory where data will be stored
            feature_store_dir = self.data_ingestion_config.data_ingestion_dir

           # create directory
            os.makedirs(feature_store_dir, exist_ok=True)

    # full file path
            feature_store_file_path = os.path.join(feature_store_dir, "data.csv")

            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")

            df.to_csv(feature_store_file_path, index=False, header=True)

            logging.info("Saved the data from S3")

            logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")

            data_ingestion_artifact = DataIngestionArtifact(ingested_file_path=feature_store_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact
         

        except Exception as e:
         raise MyException(e, sys) from e