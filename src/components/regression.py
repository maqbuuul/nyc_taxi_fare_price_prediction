import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.utils.common import save_object, validate_dataframe
from src.utils.config import Config


class FarePrediction:
    """
    Component for predicting taxi fare amounts.

    This class handles:
    1. Training a regression model
    2. Making fare predictions
    3. Evaluating prediction performance
    """

    def __init__(self):
        """Initialize FarePrediction with configuration"""
        # Load configuration
        self.config = Config()
        self.models_dir = self.config.models_dir
        self.regressor_path = os.path.join(self.models_dir, "regressor.joblib")

    def _identify_column_types(self, df):
        """
        Identify categorical and numerical columns

        Args:
            df: DataFrame to analyze

        Returns:
            Tuple of (categorical_columns, numerical_columns)
        """
        # Identify categorical columns (object type or low cardinality)
        categorical_columns = []
        numerical_columns = []

        for col in df.columns:
            # Skip the target column if it exists
            if col == "fare_amount":
                continue

            # Check if it's already a categorical or object type
            if df[col].dtype == "object" or df[col].dtype.name == "category":
                categorical_columns.append(col)
            # Check if it's a low cardinality numeric column (could be categorical)
            elif df[col].dtype.kind in "if" and df[col].nunique() < 10:
                categorical_columns.append(col)
            # Otherwise, it's numerical
            elif df[col].dtype.kind in "if":
                numerical_columns.append(col)

        logging.info(
            f"Automatically identified {len(categorical_columns)} categorical columns and {len(numerical_columns)} numerical columns"
        )
        return categorical_columns, numerical_columns

    def prepare_regression_features(self, df, preprocessor, cat_features, num_features):
        """
        Prepare features for regression

        Args:
            df: DataFrame containing trip data
            preprocessor: Fitted preprocessor
            cat_features: List of categorical features
            num_features: List of numerical features

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, preprocessor)
        """
        try:
            logging.info("Preparing features for regression")

            # Validate DataFrame
            if not validate_dataframe(df, required_columns=["fare_amount"]):
                raise ValueError("Invalid DataFrame for regression")

            # Prepare target variable
            y = df["fare_amount"]

            # Create copy of DataFrame to avoid modifying original
            X = df.copy()

            # Automatically identify column types
            categorical_cols, numerical_cols = self._identify_column_types(X)

            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            # Create preprocessing pipeline
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            numerical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Combine preprocessing steps
            feature_preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_transformer, numerical_cols),
                    ("cat", categorical_transformer, categorical_cols),
                ],
                remainder="drop",  # Drop any columns not specified
            )

            # Add cluster feature if available
            if (
                "cluster" in df.columns
                and "cluster" not in categorical_cols
                and "cluster" not in numerical_cols
            ):
                X["cluster"] = df["cluster"].astype(int)
                # Update the list of categorical columns
                categorical_cols.append("cluster")

                # Recreate feature_preprocessor with updated columns
                feature_preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", numerical_transformer, numerical_cols),
                        ("cat", categorical_transformer, categorical_cols),
                    ],
                    remainder="drop",
                )

                logging.info("Added cluster feature to regression features")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Save the preprocessor for later use
            save_object(
                os.path.join(self.models_dir, "regression_preprocessor.joblib"),
                feature_preprocessor,
            )

            logging.info(f"Features prepared successfully with shape {X.shape}")
            return X_train, X_test, y_train, y_test, feature_preprocessor

        except Exception as e:
            logging.error(f"Error preparing regression features: {e}")
            raise CustomException(e, sys)

    def train_regressor(self, X_train, y_train, preprocessor):
        """
        Train Random Forest regressor

        Args:
            X_train: Training features
            y_train: Training target values
            preprocessor: Column transformer for preprocessing

        Returns:
            Trained regressor pipeline
        """
        try:
            logging.info("Training Random Forest regressor")

            # Create a pipeline with preprocessing and regressor
            regressor_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "regressor",
                        RandomForestRegressor(
                            n_estimators=100,
                            max_depth=15,
                            min_samples_split=5,
                            random_state=42,
                        ),
                    ),
                ]
            )

            # Fit the pipeline
            regressor_pipeline.fit(X_train, y_train)

            # Try to get feature importances if possible
            try:
                # Get feature names from preprocessor
                feature_names = preprocessor.get_feature_names_out()
                # Get feature importances from regressor
                feature_importance = pd.Series(
                    regressor_pipeline.named_steps["regressor"].feature_importances_,
                    index=feature_names,
                ).sort_values(ascending=False)

                logging.info("Top 5 important features for regression:")
                for feature, importance in feature_importance.head(5).items():
                    logging.info(f"  {feature}: {importance:.4f}")
            except:
                logging.warning("Could not extract feature importances")

            # Perform cross-validation
            try:
                cv_scores = cross_val_score(
                    regressor_pipeline,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="neg_mean_squared_error",
                )
                rmse_scores = np.sqrt(-cv_scores)
                logging.info(f"Cross-validation RMSE scores: {rmse_scores}")
                logging.info(
                    f"Mean CV RMSE: {rmse_scores.mean():.2f}, Std: {rmse_scores.std():.2f}"
                )
            except:
                logging.warning("Could not perform cross-validation")

            logging.info("Regressor trained successfully")
            return regressor_pipeline

        except Exception as e:
            logging.error(f"Error training regressor: {e}")
            raise CustomException(e, sys)

    def evaluate_regressor(self, regressor, X_test, y_test):
        """
        Evaluate regressor performance

        Args:
            regressor: Trained regressor
            X_test: Test features
            y_test: Test target values

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logging.info("Evaluating regressor")

            # Make predictions
            y_pred = regressor.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Calculate percentage error
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}

            logging.info(f"Regressor evaluation completed:")
            logging.info(f"  MAE: ${mae:.2f}")
            logging.info(f"  RMSE: ${rmse:.2f}")
            logging.info(f"  R²: {r2:.3f}")
            logging.info(f"  MAPE: {mape:.2f}%")

            return metrics

        except Exception as e:
            logging.error(f"Error evaluating regressor: {e}")
            raise CustomException(e, sys)

    def initiate_regression(self, df, preprocessor, cat_features, num_features):
        """
        Entry point for regression process

        Args:
            df: DataFrame containing trip data
            preprocessor: Fitted preprocessor
            cat_features: List of categorical features
            num_features: List of numerical features

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logging.info("Starting regression process")

            # Check if DataFrame is valid
            if not validate_dataframe(df, required_columns=["fare_amount"]):
                logging.warning(
                    "Invalid DataFrame, generating synthetic data for regression"
                )
                df = self._generate_synthetic_data(100)

            # Prepare features
            X_train, X_test, y_train, y_test, feature_preprocessor = (
                self.prepare_regression_features(
                    df, preprocessor, cat_features, num_features
                )
            )

            # Train regressor
            regressor = self.train_regressor(X_train, y_train, feature_preprocessor)

            # Save regressor
            os.makedirs(self.models_dir, exist_ok=True)
            save_object(self.regressor_path, regressor)

            # Evaluate regressor
            evaluation_metrics = self.evaluate_regressor(regressor, X_test, y_test)

            logging.info("Regression process completed successfully")
            return evaluation_metrics

        except Exception as e:
            logging.error(f"Error in regression process: {e}")
            raise CustomException(e, sys)

    def _generate_synthetic_data(self, num_samples=100):
        """
        Generate synthetic data for regression

        Args:
            num_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic data
        """
        logging.info(f"Generating {num_samples} synthetic records for regression")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate data with clear relationships
        # Base fare is $2.50
        # Per mile rate is approximately $2.50
        # Per minute rate is approximately $0.50
        # Time-of-day and location factors

        trip_distances = np.random.uniform(0.5, 20, num_samples)
        trip_times = (
            trip_distances * np.random.normal(3, 0.5, num_samples) * 60
        )  # 3 minutes per mile

        # Generate passenger counts
        passenger_counts = np.random.choice(range(1, 7), num_samples)

        # Generate time of day (hour 0-23)
        pickup_hours = np.random.choice(range(24), num_samples)

        # Time of day factor (busier times cost more)
        time_factors = np.ones(num_samples)
        # Rush hours (7-9 AM, 4-7 PM) have higher rates
        time_factors[
            np.logical_or(
                np.logical_and(pickup_hours >= 7, pickup_hours <= 9),
                np.logical_and(pickup_hours >= 16, pickup_hours <= 19),
            )
        ] = 1.2
        # Late night (11 PM - 5 AM) has higher rates
        time_factors[np.logical_or(pickup_hours >= 23, pickup_hours <= 5)] = 1.3

        # Calculate fare components
        base_fare = 2.50
        distance_fare = trip_distances * 2.50
        time_fare = (trip_times / 60) * 0.50  # $0.50 per minute

        # Generate total fare with some randomness
        fare_amounts = (base_fare + distance_fare + time_fare) * time_factors
        # Add some noise
        fare_amounts = fare_amounts * np.random.normal(1, 0.1, num_samples)

        # Generate other features
        data = {
            "trip_distance": trip_distances,
            "trip_time_in_secs": trip_times,
            "passenger_count": passenger_counts,
            "pickup_hour": pickup_hours,
            "pickup_dayofweek": np.random.choice(range(7), num_samples),
            "pickup_day": np.random.choice(range(1, 29), num_samples),
            "pickup_month": np.random.choice(range(1, 13), num_samples),
            "fare_amount": fare_amounts,
            "payment_type": np.random.choice([1, 2], num_samples),
            "RatecodeID": np.random.choice(range(1, 6), num_samples),
        }

        # Add derived features
        data["speed"] = data["trip_distance"] / (data["trip_time_in_secs"] / 3600)
        data["cost_per_mile"] = data["fare_amount"] / data["trip_distance"]

        # Create clusters
        data["cluster"] = np.zeros(num_samples, dtype=int)
        # Short trips
        data["cluster"][data["trip_distance"] < 5] = 0
        # Medium trips
        data["cluster"][
            np.logical_and(data["trip_distance"] >= 5, data["trip_distance"] < 10)
        ] = 1
        # Long trips
        data["cluster"][data["trip_distance"] >= 10] = 2

        # Add categorical features
        df = pd.DataFrame(data)
        df["payment_category"] = np.random.choice(
            ["credit_card", "cash", "dispute"], num_samples
        )
        df["distance_category"] = np.random.choice(
            ["short", "medium", "long"], num_samples
        )
        df["passenger_group"] = np.random.choice(
            ["solo", "couple", "group"], num_samples
        )
        df["time_of_day"] = np.random.choice(
            ["morning", "afternoon", "evening", "night"], num_samples
        )
        df["is_weekend"] = np.random.choice([0, 1], num_samples)

        logging.info("Synthetic regression data generated successfully")
        return df
