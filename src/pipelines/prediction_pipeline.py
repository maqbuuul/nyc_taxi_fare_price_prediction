import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.utils.common import load_object, validate_dataframe
from src.utils.config import Config


class PredictionPipeline:
    """
    Pipeline for making fare predictions on new taxi trip data.

    This class loads trained models and makes predictions on new data
    using the multi-stage pipeline approach:
    1. Feature engineering
    2. Clustering
    3. Fare category classification
    4. Specialized regression based on predicted category
    """

    def __init__(self):
        """Initialize PredictionPipeline with config-based paths"""
        self.config = Config()
        self.models_dir = self.config.models_dir

        # Load models
        self.load_models()

    def load_models(self):
        """
        Load all trained models and metadata
        """
        try:
            logging.info("Loading trained models")

            # Create models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)

            # Load metadata (or create if not found)
            try:
                self.metadata = load_object(self.config.get_model_path("metadata"))
            except FileNotFoundError:
                logging.warning("Metadata file not found, creating default metadata")
                self.metadata = {
                    "cat_features": [
                        "payment_type",
                        "RatecodeID",
                        "pickup_hour",
                        "pickup_dayofweek",
                        "payment_category",
                        "distance_category",
                        "passenger_group",
                        "time_of_day",
                        "is_weekend",
                    ],
                    "num_features": [
                        "passenger_count",
                        "trip_distance",
                        "trip_time_in_secs",
                        "speed",
                        "cost_per_mile",
                        "pickup_day",
                        "pickup_month",
                    ],
                    "fare_categories": ["low", "medium", "high"],
                }
                # Save default metadata
                from src.utils.common import save_object

                save_object(self.config.get_model_path("metadata"), self.metadata)

            # Load preprocessor (or create if not found)
            try:
                self.preprocessor = load_object(
                    self.config.get_model_path("preprocessor")
                )
            except FileNotFoundError:
                logging.warning("Preprocessor not found, creating a new one")
                from src.components.data_transformation import DataTransformation

                dt = DataTransformation()
                self.preprocessor = dt.create_preprocessor(
                    self.metadata["cat_features"], self.metadata["num_features"]
                )
                # Save default preprocessor
                from src.utils.common import save_object

                save_object(
                    self.config.get_model_path("preprocessor"), self.preprocessor
                )

            # Load clusterer (or create if not found)
            try:
                self.clusterer = load_object(self.config.get_model_path("clusterer"))
            except FileNotFoundError:
                logging.warning("Clusterer not found, creating a new one")
                from sklearn.cluster import KMeans

                self.clusterer = KMeans(n_clusters=3, random_state=42)
                # Save default clusterer
                from src.utils.common import save_object

                save_object(self.config.get_model_path("clusterer"), self.clusterer)

            # Load classifier (or create if not found)
            try:
                self.classifier = load_object(self.config.get_model_path("classifier"))
            except FileNotFoundError:
                logging.warning("Classifier not found, creating a new one")
                from sklearn.ensemble import RandomForestClassifier

                self.classifier = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )
                # Save default classifier
                from src.utils.common import save_object

                save_object(self.config.get_model_path("classifier"), self.classifier)

            # Load specialized regressors (or create if not found)
            self.regressors = {}
            for category in self.metadata["fare_categories"]:
                try:
                    self.regressors[category] = load_object(
                        self.config.get_model_path(category)
                    )
                except FileNotFoundError:
                    logging.warning(
                        f"Regressor for {category} not found, creating a new one"
                    )
                    from sklearn.ensemble import RandomForestRegressor

                    self.regressors[category] = RandomForestRegressor(
                        n_estimators=100, random_state=42
                    )
                    # Save default regressor
                    from src.utils.common import save_object

                    save_object(
                        self.config.get_model_path(category), self.regressors[category]
                    )

            logging.info("All models loaded successfully")

            # Extract feature lists from metadata
            self.cat_features = self.metadata["cat_features"]
            self.num_features = self.metadata["num_features"]

        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise CustomException(e, sys)

    def preprocess_data(self, df):
        """
        Apply feature engineering to raw data

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with engineered features
        """
        try:
            logging.info("Preprocessing input data")

            # Validate input dataframe
            if not validate_dataframe(df):
                logging.warning("Invalid input DataFrame, using default values")
                df = self._create_default_dataframe()

            # Create data transformation component
            data_transformation = DataTransformation()

            # Apply feature engineering
            featured_data = data_transformation.engineer_features(df)

            logging.info("Data preprocessing completed successfully")
            return featured_data

        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise CustomException(e, sys)

    def predict_cluster(self, df):
        """
        Predict cluster for new data

        Args:
            df: DataFrame with engineered features

        Returns:
            DataFrame with cluster predictions
        """
        try:
            logging.info("Predicting clusters")

            # Select clustering features (use available ones)
            cluster_features = ["trip_distance", "cost_per_mile"]

            # Check which features are available
            available_features = [
                feature for feature in cluster_features if feature in df.columns
            ]

            # Add optional features if available
            if "pickup_hour" in df.columns:
                available_features.append("pickup_hour")
            if "passenger_count" in df.columns:
                available_features.append("passenger_count")

            # Check if we have enough features
            if len(available_features) < 2:
                logging.warning(
                    "Not enough features for clustering, adding default feature"
                )
                df["default_feature"] = 1.0
                available_features.append("default_feature")

            # Scale features
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_cluster = df[available_features].copy()
            X_cluster_scaled = scaler.fit_transform(X_cluster)

            # Predict clusters
            df_with_clusters = df.copy()
            df_with_clusters["cluster"] = self.clusterer.predict(X_cluster_scaled)

            logging.info(
                f"Cluster distribution: {df_with_clusters['cluster'].value_counts()}"
            )

            return df_with_clusters

        except Exception as e:
            logging.error(f"Error predicting clusters: {e}")
            raise CustomException(e, sys)

    def predict_fare_category(self, df):
        """
        Predict fare category for new data

        Args:
            df: DataFrame with clusters

        Returns:
            DataFrame with fare category predictions
        """
        try:
            logging.info("Predicting fare categories")

            # Prepare features for classification
            features = []

            # Find available categorical features
            for feature in self.cat_features:
                if feature in df.columns:
                    features.append(feature)

            # Find available numerical features
            for feature in self.num_features:
                if feature in df.columns:
                    features.append(feature)

            # Add cluster if available
            if "cluster" in df.columns:
                features.append("cluster")

            # Ensure we have features
            if len(features) == 0:
                logging.warning(
                    "No features available for classification, using default"
                )
                df["default_feature"] = 1.0
                features = ["default_feature"]

            # Predict fare category
            X = df[features]

            # Workaround for sklearn issue with feature names
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="mean")
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

            # Make prediction
            predicted_categories = self.classifier.predict(X_imputed)

            # Add predictions to DataFrame
            df_with_categories = df.copy()
            df_with_categories["predicted_fare_category"] = predicted_categories

            logging.info(
                f"Fare category distribution: {df_with_categories['predicted_fare_category'].value_counts()}"
            )

            return df_with_categories

        except Exception as e:
            logging.error(f"Error predicting fare categories: {e}")
            raise CustomException(e, sys)

    def predict_fare(self, df):
        """
        Predict fare using specialized regression models

        Args:
            df: DataFrame with fare categories

        Returns:
            DataFrame with fare predictions
        """
        try:
            logging.info("Predicting fares using specialized models")

            # Prepare features for regression
            features = []

            # Find available categorical features
            for feature in self.cat_features:
                if feature in df.columns:
                    features.append(feature)

            # Find available numerical features
            for feature in self.num_features:
                if feature in df.columns:
                    features.append(feature)

            # Add cluster if available
            if "cluster" in df.columns:
                features.append("cluster")

            # Ensure we have features
            if len(features) == 0:
                logging.warning("No features available for regression, using default")
                df["default_feature"] = 1.0
                features = ["default_feature"]

            # Initialize predictions array
            predictions = np.zeros(len(df))

            # For each category, use the appropriate specialized model
            for category, model in self.regressors.items():
                # Create a mask for this category
                mask = df["predicted_fare_category"] == category

                if mask.sum() > 0:
                    # Get the subset of data for this category
                    category_df = df.loc[mask, features]

                    # Handle missing values
                    from sklearn.impute import SimpleImputer

                    imputer = SimpleImputer(strategy="mean")
                    X_imputed = pd.DataFrame(
                        imputer.fit_transform(category_df), columns=category_df.columns
                    )

                    # Make predictions for this category
                    category_predictions = model.predict(X_imputed)

                    # Store predictions
                    predictions[mask] = category_predictions

                    logging.info(
                        f"Made {mask.sum()} predictions for {category} category"
                    )

            # Add predictions to DataFrame
            df_with_predictions = df.copy()
            df_with_predictions["predicted_fare"] = predictions

            # Log prediction statistics
            logging.info(
                f"Prediction statistics: min={predictions.min():.2f}, "
                f"max={predictions.max():.2f}, mean={predictions.mean():.2f}"
            )

            return df_with_predictions

        except Exception as e:
            logging.error(f"Error predicting fares: {e}")
            raise CustomException(e, sys)

    def predict(self, df):
        """
        Make end-to-end predictions on new data

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with predictions
        """
        try:
            logging.info("Starting prediction pipeline")

            # Step 1: Preprocess data
            featured_data = self.preprocess_data(df)

            # Step 2: Predict clusters
            clustered_data = self.predict_cluster(featured_data)

            # Step 3: Predict fare categories
            classified_data = self.predict_fare_category(clustered_data)

            # Step 4: Predict fares
            final_data = self.predict_fare(classified_data)

            logging.info("Prediction pipeline completed successfully")

            return final_data

        except Exception as e:
            logging.error(f"Error in prediction pipeline: {e}")
            raise CustomException(e, sys)

    def _create_default_dataframe(self):
        """Create a default DataFrame with minimal features for prediction"""
        logging.info("Creating default DataFrame for prediction")

        data = {
            "trip_distance": [5.0],
            "passenger_count": [2],
            "pickup_hour": [12],
            "payment_type": [1],  # Credit card
            "trip_time_in_secs": [900],  # 15 minutes
        }

        return pd.DataFrame(data)


class PredictionAPI:
    """
    API for making predictions on single trips or batches of trips.
    """

    def __init__(self):
        """Initialize PredictionAPI with prediction pipeline"""
        self.pipeline = PredictionPipeline()

    def predict_single_trip(self, trip_data):
        """
        Make prediction for a single trip

        Args:
            trip_data: Dictionary with trip features

        Returns:
            Dictionary with predictions
        """
        try:
            # Validate trip data
            required_fields = ["trip_distance"]
            for field in required_fields:
                if field not in trip_data:
                    logging.warning(f"Missing required field: {field}")
                    trip_data[field] = 5.0  # Default value

            # Convert single trip to DataFrame
            df = pd.DataFrame([trip_data])

            # Make prediction
            result = self.pipeline.predict(df)

            # Extract relevant information
            prediction = {
                "predicted_fare": result["predicted_fare"].iloc[0],
                "predicted_fare_category": result["predicted_fare_category"].iloc[0],
                "cluster": result["cluster"].iloc[0],
            }

            return prediction

        except Exception as e:
            logging.error(f"Error predicting single trip: {e}")
            raise CustomException(e, sys)

    def predict_batch(self, batch_data):
        """
        Make predictions for a batch of trips

        Args:
            batch_data: List of dictionaries with trip features

        Returns:
            DataFrame with predictions
        """
        try:
            # Validate batch data
            if not batch_data:
                logging.warning("Empty batch data")
                batch_data = [{"trip_distance": 5.0}]

            # Convert batch to DataFrame
            df = pd.DataFrame(batch_data)

            # Make predictions
            results = self.pipeline.predict(df)

            return results

        except Exception as e:
            logging.error(f"Error predicting batch: {e}")
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    # Create test data
    test_trip = {
        "trip_distance": 5.2,
        "passenger_count": 2,
        "pickup_hour": 18,
        "payment_type": 1,  # Credit card
    }

    # Make prediction
    api = PredictionAPI()
    result = api.predict_single_trip(test_trip)

    print(f"Predicted fare: ${result['predicted_fare']:.2f}")
    print(f"Fare category: {result['predicted_fare_category']}")
    print(f"Trip cluster: {result['cluster']}")
