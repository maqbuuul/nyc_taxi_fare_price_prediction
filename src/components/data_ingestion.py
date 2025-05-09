import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils.database import DatabaseConnector
from src.utils.config import Config


class DataIngestion:
    """
    Data ingestion component for loading and cleaning taxi trip data.

    This class handles:
    1. Loading data from the database
    2. Initial data cleaning and validation
    3. Outlier removal
    """

    def __init__(self):
        """Initialize DataIngestion with config-based paths"""
        self.config = Config()
        self.db_path = self.config.db_path

    def get_data(self, sample_size=1000):
        """
        Load data from database

        Args:
            sample_size: Number of records to load

        Returns:
            DataFrame containing the loaded data
        """
        try:
            logging.info(f"Loading {sample_size} records from all_taxi_trips")

            with DatabaseConnector(self.db_path) as db:
                df = db.get_sample("all_taxi_trips", sample_size)

            if df.empty:
                logging.warning(
                    "No data retrieved from database. Check if table exists and has data."
                )
                raise ValueError("Empty dataset returned from database query")

            logging.info(f"Successfully loaded {len(df)} records from database")
            return df

        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise CustomException(e, sys)

    def clean_data(self, df):
        """
        Clean and validate the data

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        try:
            logging.info(f"Starting data cleaning. Original shape: {df.shape}")

            # Remove records with zero or negative fares/distances
            df = df[(df["fare_amount"] > 0) & (df["trip_distance"] > 0)]
            logging.info(f"After removing zero/negative fares/distances: {df.shape}")

            # Remove fare outliers (above 99th percentile)
            fare_threshold = df["fare_amount"].quantile(0.99)
            df = df[df["fare_amount"] <= fare_threshold]
            logging.info(f"After removing fare outliers: {df.shape}")

            # Remove distance outliers (above 99th percentile)
            distance_threshold = df["trip_distance"].quantile(0.99)
            df = df[df["trip_distance"] <= distance_threshold]
            logging.info(f"After removing distance outliers: {df.shape}")

            # Final validation
            if df.empty:
                logging.error(
                    "After cleaning, the dataset is empty. Check your filters or data source."
                )
                raise ValueError("Empty dataset after cleaning")

            logging.info(
                f"Data cleaning completed successfully. Final shape: {df.shape}"
            )
            return df

        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self, sample_size=1000, use_synthetic=False):
        """
        Initiate the data ingestion process

        Args:
            sample_size: Number of records to load
            use_synthetic: Whether to use synthetic data instead of database data

        Returns:
            Cleaned DataFrame
        """
        try:
            # Decide whether to use real or synthetic data
            if use_synthetic:
                logging.info("Using synthetic data as requested")
                df = self._generate_synthetic_data(sample_size)
            else:
                # Load data from database
                logging.info("Using real data from database")
                df = self.get_data(sample_size)

            # Clean data
            df = self.clean_data(df)

            logging.info("Data ingestion completed successfully")
            return df

        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise CustomException(e, sys)

    def _generate_synthetic_data(self, num_samples=1000):
        """
        Generate synthetic data for testing or when database data is unavailable

        Args:
            num_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic data
        """
        logging.info(f"Generating {num_samples} synthetic records")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate data
        data = {
            "id": range(1, num_samples + 1),
            "VendorID": np.random.choice([1, 2], num_samples),
            "tpep_pickup_datetime": pd.date_range(
                start="2024-01-01", periods=num_samples, freq="H"
            ),
            "tpep_dropoff_datetime": pd.date_range(
                start="2024-01-01 01:00:00", periods=num_samples, freq="H"
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

        # Calculate trip time in seconds
        trip_times = []
        for i in range(num_samples):
            pickup = data["tpep_pickup_datetime"][i]
            dropoff = data["tpep_dropoff_datetime"][i]
            delta = dropoff - pickup
            trip_times.append(delta.total_seconds())

        data["trip_time_in_secs"] = trip_times

        # Create DataFrame
        df = pd.DataFrame(data)

        logging.info("Synthetic data generated successfully")
        return df
