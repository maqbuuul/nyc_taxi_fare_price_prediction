import os
import sys
import yaml
from src.logger import logging
from src.exception import CustomException


class Config:
    """
    Configuration class for managing project settings.

    This class:
    1. Loads configuration from YAML file
    2. Provides access to configuration parameters
    3. Manages file paths
    """

    def __init__(self, config_path=None):
        """
        Initialize the Config class

        Args:
            config_path: Path to configuration file (optional)
        """
        try:
            # Set project root directory
            self.root_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            # Set default config path if not provided
            if config_path is None:
                config_path = os.path.join(self.root_dir, "config.yaml")

            # Load configuration
            self.config = self._load_config(config_path)

            # Set up directories
            self.setup_directories()

            # Set up attributes
            self._initialize_attributes()

        except Exception as e:
            logging.error(f"Error initializing configuration: {e}")
            raise CustomException(e, sys)

    def _load_config(self, config_path):
        """
        Load configuration from YAML file

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary with configuration
        """
        try:
            # Check if configuration file exists
            if not os.path.exists(config_path):
                logging.warning(
                    f"Configuration file not found at {config_path}. Using default settings."
                )
                return self._default_config()

            # Load configuration
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)

            logging.info(f"Configuration loaded from {config_path}")
            return config

        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            logging.warning("Using default configuration.")
            return self._default_config()

    def _default_config(self):
        """
        Create default configuration

        Returns:
            Dictionary with default configuration
        """
        return {
            "DB_PATH": os.path.join(self.root_dir, "data", "nyc_taxi.duckdb"),
            "SAMPLE_SIZE": 100000,
            "FARE_OUTLIER_QUANTILES": [0.01, 0.99],
            "DISTANCE_OUTLIER_QUANTILES": [0.01, 0.99],
            "CLUSTER_FEATURES": ["trip_distance", "fare_amount", "cost_per_mile"],
            "CLUSTER_RANGE": list(range(2, 11)),
            "CLUSTER_RANDOM_STATE": 42,
            "CLASSIFICATION_RANDOM_STATE": 42,
            "CLASSIFICATION_TEST_SIZE": 0.2,
            "CAT_FEATURES": [
                "payment_type",
                "RatecodeID",
                "pickup_hour",
                "pickup_dayofweek",
            ],
            "NUM_FEATURES": [
                "passenger_count",
                "trip_distance",
                "trip_time_in_secs",
                "speed",
                "fare_per_mile",
                "pickup_day",
                "pickup_month",
            ],
            "REGRESSION_RANDOM_STATE": 42,
            "REGRESSION_TEST_SIZE": 0.2,
            "CV_FOLDS": 5,
            "FARE_CATEGORIES": ["low", "medium", "high"],
            "MODEL_DIR": os.path.join(self.root_dir, "models"),
        }

    def setup_directories(self):
        """Create necessary directories"""
        try:
            # Create data directory
            data_dir = os.path.join(self.root_dir, "data")
            os.makedirs(data_dir, exist_ok=True)

            # Create models directory
            model_dir = self.config.get(
                "MODEL_DIR", os.path.join(self.root_dir, "models")
            )
            os.makedirs(model_dir, exist_ok=True)

            # Create logs directory
            logs_dir = os.path.join(self.root_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)

            logging.info("Directory structure set up successfully")

        except Exception as e:
            logging.error(f"Error setting up directories: {e}")
            raise CustomException(e, sys)

    def _initialize_attributes(self):
        """Initialize attributes from configuration"""
        try:
            # Database settings
            self.db_path = self.config.get(
                "DB_PATH", os.path.join(self.root_dir, "data", "nyc_taxi.duckdb")
            )
            self.sample_size = self.config.get("SAMPLE_SIZE", 1000)

            # Outlier settings
            self.fare_outlier_quantiles = self.config.get(
                "FARE_OUTLIER_QUANTILES", [0.01, 0.99]
            )
            self.distance_outlier_quantiles = self.config.get(
                "DISTANCE_OUTLIER_QUANTILES", [0.01, 0.99]
            )

            # Clustering settings
            self.cluster_features = self.config.get(
                "CLUSTER_FEATURES", ["trip_distance", "fare_amount", "cost_per_mile"]
            )
            self.cluster_range = self.config.get("CLUSTER_RANGE", list(range(2, 11)))
            self.cluster_random_state = self.config.get("CLUSTER_RANDOM_STATE", 42)

            # Classification settings
            self.classification_random_state = self.config.get(
                "CLASSIFICATION_RANDOM_STATE", 42
            )
            self.classification_test_size = self.config.get(
                "CLASSIFICATION_TEST_SIZE", 0.2
            )

            # Feature settings
            self.cat_features = self.config.get("CAT_FEATURES", [])
            self.num_features = self.config.get("NUM_FEATURES", [])

            # Regression settings
            self.regression_random_state = self.config.get(
                "REGRESSION_RANDOM_STATE", 42
            )
            self.regression_test_size = self.config.get("REGRESSION_TEST_SIZE", 0.2)

            # Cross-validation settings
            self.cv_folds = self.config.get("CV_FOLDS", 5)

            # Fare categories
            self.fare_categories = self.config.get(
                "FARE_CATEGORIES", ["low", "medium", "high"]
            )

            # Model directory
            self.models_dir = self.config.get(
                "MODEL_DIR", os.path.join(self.root_dir, "models")
            )

        except Exception as e:
            logging.error(f"Error initializing attributes: {e}")
            raise CustomException(e, sys)

    def get_model_path(self, model_name):
        """
        Get the path for a model file

        Args:
            model_name: Name of the model

        Returns:
            Path to the model file
        """
        return os.path.join(self.models_dir, f"{model_name}.joblib")
