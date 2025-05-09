import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException
from src.utils.common import save_object, validate_dataframe
from src.utils.config import Config


class DataTransformation:
    """
    Component for feature engineering and preprocessing.

    This class handles creating new features, preprocessing the data,
    and preparing it for model training.
    """

    def __init__(self):
        """Initialize DataTransformation with configuration"""
        # Load configuration
        self.config = Config()
        self.models_dir = self.config.models_dir
        self.preprocessor_path = os.path.join(self.models_dir, "preprocessor.joblib")

    def engineer_features(self, df):
        """
        Create new features from raw data

        Args:
            df: DataFrame containing raw data

        Returns:
            DataFrame with new features
        """
        try:
            logging.info("Creating new features")

            # Check if DataFrame is valid
            if not validate_dataframe(df):
                logging.warning("Invalid DataFrame for feature engineering")
                return df

            # Create a copy of the DataFrame
            df_featured = df.copy()

            # Time-based features
            if "tpep_pickup_datetime" in df.columns:
                # Convert to datetime if string
                if isinstance(df["tpep_pickup_datetime"].iloc[0], str):
                    df_featured["tpep_pickup_datetime"] = pd.to_datetime(
                        df["tpep_pickup_datetime"]
                    )

                # Extract time features
                df_featured["pickup_hour"] = df_featured["tpep_pickup_datetime"].dt.hour
                df_featured["pickup_day"] = df_featured["tpep_pickup_datetime"].dt.day
                df_featured["pickup_month"] = df_featured[
                    "tpep_pickup_datetime"
                ].dt.month
                df_featured["pickup_dayofweek"] = df_featured[
                    "tpep_pickup_datetime"
                ].dt.dayofweek
                df_featured["is_weekend"] = (
                    df_featured["pickup_dayofweek"].isin([5, 6]).astype(int)
                )

                # Time of day category
                time_bins = [0, 6, 12, 18, 24]
                time_labels = ["night", "morning", "afternoon", "evening"]
                df_featured["time_of_day"] = pd.cut(
                    df_featured["pickup_hour"],
                    bins=time_bins,
                    labels=time_labels,
                    include_lowest=True,
                )

                logging.info("Time-based features created")

            # Distance-based features
            if all(col in df.columns for col in ["trip_distance", "trip_time_in_secs"]):
                # Calculate speed (mph)
                df_featured["speed"] = df_featured["trip_distance"] / (
                    df_featured["trip_time_in_secs"] / 3600
                )

                # Handle infinite values and NaNs
                df_featured["speed"] = df_featured["speed"].replace(
                    [np.inf, -np.inf], np.nan
                )
                df_featured["speed"] = df_featured["speed"].fillna(
                    df_featured["speed"].median()
                )

                # Distance category
                distance_bins = [0, 2, 5, 10, float("inf")]
                distance_labels = ["very_short", "short", "medium", "long"]
                df_featured["distance_category"] = pd.cut(
                    df_featured["trip_distance"],
                    bins=distance_bins,
                    labels=distance_labels,
                    include_lowest=True,
                )

                logging.info("Distance-based features created")

            # Payment and fare features
            if all(col in df.columns for col in ["fare_amount", "trip_distance"]):
                # Calculate cost per mile
                df_featured["cost_per_mile"] = (
                    df_featured["fare_amount"] / df_featured["trip_distance"]
                )

                # Handle infinite values and NaNs
                df_featured["cost_per_mile"] = df_featured["cost_per_mile"].replace(
                    [np.inf, -np.inf], np.nan
                )
                df_featured["cost_per_mile"] = df_featured["cost_per_mile"].fillna(
                    df_featured["cost_per_mile"].median()
                )

                logging.info("Payment and fare features created")

            # Passenger features
            if "passenger_count" in df.columns:
                # Passenger group category
                df_featured["passenger_group"] = "solo"
                df_featured.loc[
                    df_featured["passenger_count"] == 2, "passenger_group"
                ] = "couple"
                df_featured.loc[
                    df_featured["passenger_count"] > 2, "passenger_group"
                ] = "group"

                logging.info("Passenger features created")

            # Payment type category
            if "payment_type" in df.columns:
                # Map payment types
                payment_map = {
                    1: "credit_card",
                    2: "cash",
                    3: "no_charge",
                    4: "dispute",
                    5: "unknown",
                    6: "voided",
                }
                df_featured["payment_category"] = df_featured["payment_type"].map(
                    payment_map
                )

                # Handle missing values
                df_featured["payment_category"] = df_featured[
                    "payment_category"
                ].fillna("unknown")

                logging.info("Payment type category created")

            logging.info("Feature engineering completed successfully")
            return df_featured

        except Exception as e:
            logging.error(f"Error creating features: {e}")
            raise CustomException(e, sys)

    def create_preprocessor(self, cat_features, num_features):
        """
        Create preprocessing pipeline

        Args:
            cat_features: List of categorical feature names
            num_features: List of numerical feature names

        Returns:
            Fitted preprocessor
        """
        try:
            logging.info("Creating preprocessing pipeline")

            # Create preprocessing steps
            numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

            categorical_transformer = Pipeline(
                steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
            )

            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, num_features),
                    ("cat", categorical_transformer, cat_features),
                ],
                remainder="passthrough",  # Include any other columns
            )

            logging.info("Preprocessor created successfully")
            return preprocessor

        except Exception as e:
            logging.error(f"Error creating preprocessor: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, df):
        """
        Entry point for data transformation

        Args:
            df: DataFrame to transform

        Returns:
            Tuple of (transformed DataFrame, preprocessor, categorical features, numerical features)
        """
        try:
            logging.info("Starting data transformation")

            # Check if DataFrame is empty
            if df.empty:
                logging.warning(
                    "DataFrame is empty, generating synthetic data for transformation"
                )
                df = self._generate_synthetic_data(100)

            # Create features
            df = self.engineer_features(df)

            # Define feature lists
            cat_features = [
                "payment_type",
                "RatecodeID",
                "pickup_hour",
                "pickup_dayofweek",
                "payment_category",
                "distance_category",
                "passenger_group",
                "time_of_day",
                "is_weekend",
            ]

            num_features = [
                "passenger_count",
                "trip_distance",
                "trip_time_in_secs",
                "speed",
                "cost_per_mile",
                "pickup_day",
                "pickup_month",
            ]

            # Filter for columns that exist
            cat_features = [col for col in cat_features if col in df.columns]
            num_features = [col for col in num_features if col in df.columns]

            logging.info(f"Selected categorical features: {cat_features}")
            logging.info(f"Selected numerical features: {num_features}")

            # Create and fit preprocessor
            preprocessor = self.create_preprocessor(cat_features, num_features)

            # Save preprocessor
            os.makedirs(self.models_dir, exist_ok=True)
            save_object(self.preprocessor_path, preprocessor)

            logging.info("Data transformation completed successfully")
            return df, preprocessor, cat_features, num_features

        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)

    def _generate_synthetic_data(self, num_samples=100):
        """
        Generate synthetic data for transformation

        Args:
            num_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic data
        """
        logging.info(f"Generating {num_samples} synthetic records for transformation")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate data
        data = {
            "VendorID": np.random.choice([1, 2], num_samples),
            "tpep_pickup_datetime": pd.date_range(
                start="2023-01-01", periods=num_samples, freq="H"
            ),
            "tpep_dropoff_datetime": pd.date_range(
                start="2023-01-01 01:00:00", periods=num_samples, freq="H"
            ),
            "passenger_count": np.random.choice(range(1, 7), num_samples),
            "trip_distance": np.random.uniform(0.5, 20, num_samples),
            "RatecodeID": np.random.choice(range(1, 7), num_samples),
            "store_and_fwd_flag": np.random.choice(["Y", "N"], num_samples),
            "PULocationID": np.random.randint(1, 265, num_samples),
            "DOLocationID": np.random.randint(1, 265, num_samples),
            "payment_type": np.random.choice([1, 2, 3, 4], num_samples),
            "fare_amount": np.random.uniform(2.5, 100, num_samples),
            "extra": np.random.uniform(0, 5, num_samples),
            "mta_tax": np.random.uniform(0, 1, num_samples),
            "tip_amount": np.random.uniform(0, 20, num_samples),
            "tolls_amount": np.random.uniform(0, 10, num_samples),
            "improvement_surcharge": np.random.uniform(0, 1, num_samples),
            "total_amount": np.random.uniform(5, 150, num_samples),
        }

        # Calculate trip time in seconds (approximately 1 minute per mile on average, with variation)
        trip_times = []
        for distance in data["trip_distance"]:
            # Average speed 20 mph with some variation
            avg_speed = np.random.uniform(10, 30)
            # Time in hours = distance / speed
            time_hours = distance / avg_speed
            # Convert to seconds
            time_secs = int(time_hours * 3600)
            trip_times.append(time_secs)

        data["trip_time_in_secs"] = trip_times

        # Create DataFrame
        df = pd.DataFrame(data)

        logging.info("Synthetic transformation data generated successfully")
        return df
