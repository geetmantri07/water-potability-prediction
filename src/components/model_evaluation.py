from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ModelEvaluationArtifact
from sklearn.metrics import accuracy_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import numpy as np
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
import os
from dataclasses import dataclass


import mlflow
import mlflow.sklearn


from dotenv import load_dotenv



load_dotenv(override=True)


@dataclass
class EvaluateModelResponse:
    trained_model_accuracy: float
    best_model_accuracy: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact

            mlflow.set_tracking_uri(
                "https://dagshub.com/geetmantri07/water-potability-prediction.mlflow"
            )

            # Set credentials from .env
            os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

            # ✅ Set MLflow experiment
            mlflow.set_experiment("my_pipeline")

        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path

            proj1_estimator = Proj1Estimator(
                bucket_name=bucket_name,
                model_path=model_path
            )

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None

        except Exception as e:
            raise MyException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            # ✅ Start MLflow run
            with mlflow.start_run():

                # ✅ Load transformed test data
                test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path)

                feature_names = [
                    'ph', 'Hardness', 'Solids', 'Chloramines',
                    'Sulfate', 'Conductivity', 'Organic_carbon',
                    'Trihalomethanes', 'Turbidity'
                ]

                x = pd.DataFrame(test_arr[:, :-1], columns=feature_names)
                y = test_arr[:, -1]

                logging.info("Test data loaded (already transformed).")

                # ✅ Load trained model
                trained_model = load_object(
                    file_path=self.model_trainer_artifact.trained_model_file_path
                )

                logging.info("Trained model loaded.")

                # ✅ Evaluate trained model
                y_hat_trained = trained_model.predict(x)
                trained_model_accuracy = accuracy_score(y, y_hat_trained)

                logging.info(f"Accuracy (New Model): {trained_model_accuracy}")

                # ✅ Compare with production model
                best_model_accuracy = None
                best_model = self.get_best_model()

                if best_model is not None:
                    logging.info("Computing Accuracy for production model..")

                    y_hat_best_model = best_model.predict(x)
                    best_model_accuracy = accuracy_score(y, y_hat_best_model)

                    logging.info(
                        f"Accuracy-Production Model: {best_model_accuracy}, "
                        f"Accuracy-New Model: {trained_model_accuracy}"
                    )

                tmp_best_model_score = 0 if best_model_accuracy is None else best_model_accuracy
                is_model_accepted = trained_model_accuracy > tmp_best_model_score

                # ✅ MLflow Logging
                # ✅ MLflow Logging
                mlflow.log_metric("accuracy", trained_model_accuracy)

                if best_model_accuracy is not None:
                 mlflow.log_metric("best_model_accuracy", best_model_accuracy)
                 
                model = trained_model.trained_model_object

                # ✅ Log model parameters (clean & explicit)
                mlflow.log_params({
                "model_type": "RandomForest",
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
                "min_samples_leaf": model.min_samples_leaf,
                "max_features": model.max_features})

                # ✅ Tagging (bonus)
                mlflow.set_tag(
                    "stage",
                    "production" if is_model_accepted else "staging"
                )

                # 🔥 Register ONLY if accepted
                if is_model_accepted:
                    mlflow.sklearn.log_model(
                        trained_model,
                        artifact_path="model",
                        registered_model_name="WaterPotabilityModel"
                    )

                    logging.info("Model registered in MLflow (best model).")

                result = EvaluateModelResponse(
                    trained_model_accuracy=trained_model_accuracy,
                    best_model_accuracy=best_model_accuracy,
                    is_model_accepted=is_model_accepted,
                    difference=trained_model_accuracy - tmp_best_model_score
                )

                logging.info(f"Result: {result}")

                return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")

            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")

            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys) from e