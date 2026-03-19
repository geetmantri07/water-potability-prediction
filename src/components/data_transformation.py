import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from src.constants import TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data


class DataTransformation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise MyException(e, sys)


    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)


    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates preprocessing pipeline with:
        1. KNNImputer for handling missing values
        2. StandardScaler for feature scaling
        """

        logging.info("Entered get_data_transformer_object method")

        try:

            preprocessing_pipeline = Pipeline(
                steps=[
                    ("imputer", KNNImputer(n_neighbors=5)),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Pipeline created successfully: KNNImputer + StandardScaler")

            return preprocessing_pipeline

        except Exception as e:
            raise MyException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates data transformation process
        """

        try:
            logging.info("Data Transformation Started")

            # Check if data validation passed
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Data validation successful. Proceeding with transformation")

            # Load dataset
            data_path = self.data_ingestion_artifact.ingested_file_path

            df = self.read_data(data_path)

            logging.info("Dataset loaded successfully")

            # Split features and target
            input_feature_df = df.drop(columns=[TARGET_COLUMN])
            target_feature_df = df[TARGET_COLUMN]

            logging.info("Separated input features and target column")

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                input_feature_df,
                target_feature_df,
                test_size=0.2,
                random_state=42
            )

            logging.info("Train-test split completed")

            # Get preprocessing pipeline
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing on training data")

            X_train_arr = preprocessing_obj.fit_transform(X_train)

            logging.info("Applying preprocessing on test data")

            X_test_arr = preprocessing_obj.transform(X_test)

            # Combine features and target
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            logging.info("Feature and target arrays concatenated")

            # Save preprocessing object
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessing_obj
            )

            logging.info("Preprocessing object saved")

            # Save transformed train data
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )

            # Save transformed test data
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            logging.info("Transformed train and test arrays saved")

            logging.info("Data Transformation completed successfully")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys)