import sys
from src.entity.config_entity import WaterPredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class WaterData:
    def __init__(self,
                 ph,
                 Hardness,
                 Solids,
                 Chloramines,
                 Sulfate,
                 Conductivity,
                 Organic_carbon,
                 Trihalomethanes,
                 Turbidity
                 ):
        """
        Water Data constructor
        Input: all features for prediction
        """
        try:
            self.ph = ph
            self.Hardness = Hardness
            self.Solids = Solids
            self.Chloramines = Chloramines
            self.Sulfate = Sulfate
            self.Conductivity = Conductivity
            self.Organic_carbon = Organic_carbon
            self.Trihalomethanes = Trihalomethanes
            self.Turbidity = Turbidity

        except Exception as e:
            raise MyException(e, sys) from e

    def get_water_input_data_frame(self) -> DataFrame:
        """
        Returns DataFrame for prediction
        """
        try:
            water_input_dict = self.get_water_data_as_dict()
            return DataFrame(water_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e


    def get_water_data_as_dict(self):
        """
        Returns input data as dictionary
        """
        logging.info("Entered get_water_data_as_dict method")

        try:
            input_data = {
                "ph": [self.ph],
                "Hardness": [self.Hardness],
                "Solids": [self.Solids],
                "Chloramines": [self.Chloramines],
                "Sulfate": [self.Sulfate],
                "Conductivity": [self.Conductivity],
                "Organic_carbon": [self.Organic_carbon],
                "Trihalomethanes": [self.Trihalomethanes],
                "Turbidity": [self.Turbidity]
            }

            logging.info("Created water data dict")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e


class WaterDataClassifier:
    def __init__(self,
                 prediction_pipeline_config: WaterPredictorConfig = WaterPredictorConfig(),
                 ) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        """
        Returns prediction
        """
        try:
            logging.info("Entered predict method of WaterDataClassifier")

            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )

            result = model.predict(dataframe)

            return result

        except Exception as e:
            raise MyException(e, sys)