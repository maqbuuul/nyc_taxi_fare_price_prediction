import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.clustering import TripClustering
from src.components.classification import FareCategoryClassification
from src.components.regression import FarePrediction
from src.utils.config import Config


class TrainingPipeline:
    """
    Pipeline for training all models in the system.

    This pipeline orchestrates:
    1. Data ingestion and cleaning
    2. Feature engineering and preprocessing
    3. Trip clustering
    4. Fare category classification
    5. Fare amount prediction
    """

    def __init__(self):
        """Initialize TrainingPipeline with configuration"""
        # Load configuration
        self.config = Config()
        self.models_dir = self.config.models_dir

        # Create necessary directories
        os.makedirs(self.models_dir, exist_ok=True)

        # Save metadata
        self._save_metadata()

    def _save_metadata(self):
        """Save model metadata"""
        try:
            from src.utils.common import save_object
            import joblib

            metadata = {
                "cat_features": self.config.cat_features,
                "num_features": self.config.num_features,
                "fare_categories": self.config.fare_categories,
            }

            metadata_path = os.path.join(self.models_dir, "metadata.joblib")
            save_object(metadata_path, metadata)

            logging.info("Model metadata saved successfully")

        except Exception as e:
            logging.error(f"Error saving metadata: {e}")
            raise CustomException(e, sys)

    def run_pipeline(self, sample_size=None, use_synthetic=False):
        """
        Run the complete training pipeline

        Args:
            sample_size: Number of samples to use for training
            use_synthetic: Whether to use synthetic data instead of database data

        Returns:
            Dictionary with evaluation metrics for all models
        """
        try:
            logging.info("Starting training pipeline")

            # Use configured sample size if not provided
            if sample_size is None:
                sample_size = self.config.sample_size

            # Step 1: Data Ingestion
            data_ingestion = DataIngestion()
            df = data_ingestion.initiate_data_ingestion(
                sample_size, use_synthetic=use_synthetic
            )
            logging.info("Data ingestion completed")

            # Check if DataFrame is empty
            if df.empty:
                logging.error(
                    "Data ingestion returned an empty DataFrame. Cannot proceed with training."
                )
                raise CustomException("Empty DataFrame after data ingestion", sys)

            # Step 2: Data Transformation
            data_transformation = DataTransformation()
            df, preprocessor, cat_features, num_features = (
                data_transformation.initiate_data_transformation(df)
            )
            logging.info("Data transformation completed")

            # Step 3: Trip Clustering
            clustering = TripClustering()
            cluster_labels, cluster_stats = clustering.initiate_clustering(df)
            logging.info("Trip clustering completed")

            # Step 4: Fare Category Classification
            classification = FareCategoryClassification()
            classification_metrics = classification.initiate_classification(
                df, preprocessor, cat_features, num_features
            )
            logging.info("Fare category classification completed")

            # Step 5: Fare Prediction
            regression = FarePrediction()
            regression_metrics = regression.initiate_regression(
                df, preprocessor, cat_features, num_features
            )
            logging.info("Fare prediction completed")

            # Compile all results
            results = {
                "clustering": {"cluster_stats": cluster_stats.to_dict()},
                "classification": classification_metrics,
                "regression": regression_metrics,
            }

            logging.info("Training pipeline completed successfully")
            return results

        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys)
