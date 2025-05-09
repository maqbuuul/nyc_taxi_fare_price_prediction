import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.utils.common import save_object, validate_dataframe
from src.utils.config import Config


class FareCategoryClassification:
    """
    Component for classifying taxi trips into fare categories.

    This class handles:
    1. Creating fare categories
    2. Training a classifier
    3. Evaluating classification performance
    """

    def __init__(self):
        """Initialize FareCategoryClassification with configuration"""
        # Load configuration
        self.config = Config()
        self.models_dir = self.config.models_dir
        self.classifier_path = os.path.join(self.models_dir, "classifier.joblib")

    def create_fare_categories(self, df):
        """
        Create fare categories based on fare amounts

        Args:
            df: DataFrame containing fare amounts

        Returns:
            Series with fare categories
        """
        try:
            logging.info("Creating fare categories")

            # Check if DataFrame is empty or missing fare_amount
            if not validate_dataframe(df, required_columns=["fare_amount"]):
                logging.warning(
                    "Invalid DataFrame, generating synthetic fare categories"
                )
                return pd.Series(["medium"] * len(df))

            # Define fare categories using quantiles
            q1 = df["fare_amount"].quantile(0.33)
            q2 = df["fare_amount"].quantile(0.66)

            # Create categories
            categories = pd.cut(
                df["fare_amount"],
                bins=[-np.inf, q1, q2, np.inf],
                labels=["low", "medium", "high"],
            )

            logging.info(
                f"Fare categories created successfully: low < ${q1:.2f} <= medium < ${q2:.2f} <= high"
            )
            return categories

        except Exception as e:
            logging.error(f"Error creating fare categories: {e}")
            raise CustomException(e, sys)

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

    def prepare_classification_features(
        self, df, preprocessor, cat_features, num_features
    ):
        """
        Prepare features for classification

        Args:
            df: DataFrame containing trip data
            preprocessor: Fitted preprocessor
            cat_features: List of categorical features
            num_features: List of numerical features

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            logging.info("Preparing features for classification")

            # Create fare categories
            y = self.create_fare_categories(df)

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
            preprocessor = ColumnTransformer(
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
                # Update preprocessor to include cluster
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", numerical_transformer, numerical_cols + ["cluster"]),
                        ("cat", categorical_transformer, categorical_cols),
                    ],
                    remainder="drop",
                )
                logging.info("Added cluster feature to classification features")

            # Split data first
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Save the preprocessor for later use
            save_object(
                os.path.join(self.models_dir, "classification_preprocessor.joblib"),
                preprocessor,
            )

            logging.info(f"Features prepared successfully with shape {X.shape}")
            return X_train, X_test, y_train, y_test, preprocessor

        except Exception as e:
            logging.error(f"Error preparing classification features: {e}")
            raise CustomException(e, sys)

    def train_classifier(self, X_train, y_train, preprocessor):
        """
        Train Random Forest classifier

        Args:
            X_train: Training features
            y_train: Training labels
            preprocessor: Column transformer for preprocessing

        Returns:
            Trained classifier pipeline
        """
        try:
            logging.info("Training Random Forest classifier")

            # Create a pipeline with preprocessing and classifier
            classifier_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=100,
                            max_depth=10,
                            min_samples_split=5,
                            random_state=42,
                        ),
                    ),
                ]
            )

            # Fit the pipeline
            classifier_pipeline.fit(X_train, y_train)

            # Try to get feature importances if possible
            try:
                # Get feature names from preprocessor
                feature_names = preprocessor.get_feature_names_out()
                # Get feature importances from classifier
                feature_importance = pd.Series(
                    classifier_pipeline.named_steps["classifier"].feature_importances_,
                    index=feature_names,
                ).sort_values(ascending=False)

                logging.info("Top 5 important features for classification:")
                for feature, importance in feature_importance.head(5).items():
                    logging.info(f"  {feature}: {importance:.4f}")
            except:
                logging.warning("Could not extract feature importances")

            logging.info("Classifier trained successfully")
            return classifier_pipeline

        except Exception as e:
            logging.error(f"Error training classifier: {e}")
            raise CustomException(e, sys)

    def evaluate_classifier(self, classifier, X_test, y_test):
        """
        Evaluate classifier performance

        Args:
            classifier: Trained classifier
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logging.info("Evaluating classifier")

            # Make predictions
            y_pred = classifier.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            logging.info(f"Classifier evaluation completed. Accuracy: {accuracy:.3f}")
            return {"accuracy": accuracy, "classification_report": report}

        except Exception as e:
            logging.error(f"Error evaluating classifier: {e}")
            raise CustomException(e, sys)

    def initiate_classification(self, df, preprocessor, cat_features, num_features):
        """
        Entry point for classification process

        Args:
            df: DataFrame containing trip data
            preprocessor: Fitted preprocessor
            cat_features: List of categorical features
            num_features: List of numerical features

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logging.info("Starting classification process")

            # Check if DataFrame is valid
            if not validate_dataframe(df):
                logging.warning(
                    "Invalid DataFrame, generating synthetic data for classification"
                )
                df = self._generate_synthetic_data(100)

            # Prepare features
            X_train, X_test, y_train, y_test, feature_preprocessor = (
                self.prepare_classification_features(
                    df, preprocessor, cat_features, num_features
                )
            )

            # Train classifier
            classifier = self.train_classifier(X_train, y_train, feature_preprocessor)

            # Save classifier
            os.makedirs(self.models_dir, exist_ok=True)
            save_object(self.classifier_path, classifier)

            # Train and save specialized models for each fare category
            self._train_specialized_models(
                df, feature_preprocessor, cat_features, num_features
            )

            # Evaluate classifier
            evaluation_metrics = self.evaluate_classifier(classifier, X_test, y_test)

            logging.info("Classification process completed successfully")
            return evaluation_metrics

        except Exception as e:
            logging.error(f"Error in classification process: {e}")
            raise CustomException(e, sys)

    def _train_specialized_models(
        self, df, feature_preprocessor, cat_features, num_features
    ):
        """
        Train specialized models for each fare category

        Args:
            df: DataFrame containing trip data
            feature_preprocessor: Preprocessor for feature transformation
            cat_features: List of categorical features
            num_features: List of numerical features
        """
        try:
            logging.info("Training specialized models for each fare category")

            # Create fare categories
            df = df.copy()
            df["fare_category"] = self.create_fare_categories(df)

            # For each category
            for category in ["low", "medium", "high"]:
                try:
                    # Filter data for this category
                    category_df = df[df["fare_category"] == category]

                    # Skip if category is empty
                    if len(category_df) < 10:
                        logging.warning(
                            f"Not enough data for category '{category}', skipping specialized model"
                        )
                        continue

                    # Identify column types
                    categorical_cols, numerical_cols = self._identify_column_types(
                        category_df
                    )

                    # Prepare target
                    X = category_df
                    y = category_df["fare_amount"]

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    # Create a regressor pipeline with preprocessing
                    from sklearn.ensemble import RandomForestRegressor

                    regressor_pipeline = Pipeline(
                        steps=[
                            ("preprocessor", feature_preprocessor),
                            (
                                "regressor",
                                RandomForestRegressor(
                                    n_estimators=100, max_depth=10, random_state=42
                                ),
                            ),
                        ]
                    )

                    # Train regressor
                    regressor_pipeline.fit(X_train, y_train)

                    # Save regressor
                    save_object(
                        os.path.join(self.models_dir, f"{category}.joblib"),
                        regressor_pipeline,
                    )

                    # Evaluate
                    from sklearn.metrics import mean_squared_error, r2_score

                    y_pred = regressor_pipeline.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)

                    logging.info(
                        f"Specialized model for '{category}' category trained. RMSE: {rmse:.2f}, R²: {r2:.3f}"
                    )

                except Exception as e:
                    logging.error(
                        f"Error training specialized model for '{category}': {e}"
                    )
                    # Continue with other categories

        except Exception as e:
            logging.error(f"Error training specialized models: {e}")
            raise CustomException(e, sys)

    def _generate_synthetic_data(self, num_samples=100):
        """
        Generate synthetic data for classification

        Args:
            num_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic data
        """
        logging.info(f"Generating {num_samples} synthetic records for classification")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate data with clear category structure
        data = {
            "trip_distance": np.concatenate(
                [
                    np.random.normal(3, 1, num_samples // 3),
                    np.random.normal(8, 1.5, num_samples // 3),
                    np.random.normal(15, 2, num_samples // 3),
                ]
            ),
            "fare_amount": np.concatenate(
                [
                    np.random.normal(10, 2, num_samples // 3),
                    np.random.normal(25, 5, num_samples // 3),
                    np.random.normal(45, 7, num_samples // 3),
                ]
            ),
            "passenger_count": np.random.choice(range(1, 7), num_samples),
            "payment_type": np.random.choice([1, 2, 3, 4], num_samples),
            "RatecodeID": np.random.choice(range(1, 7), num_samples),
            "pickup_hour": np.random.choice(range(24), num_samples),
            "pickup_dayofweek": np.random.choice(range(7), num_samples),
            "pickup_day": np.random.choice(range(1, 29), num_samples),
            "pickup_month": np.random.choice(range(1, 13), num_samples),
            "cluster": np.random.choice(range(3), num_samples),
        }

        # Calculate derived features
        data["trip_time_in_secs"] = data["trip_distance"] * np.random.normal(
            180, 30, num_samples
        )
        data["speed"] = data["trip_distance"] / (data["trip_time_in_secs"] / 3600)
        data["cost_per_mile"] = data["fare_amount"] / data["trip_distance"]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Add categorical features
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

        logging.info("Synthetic classification data generated successfully")
        return df
