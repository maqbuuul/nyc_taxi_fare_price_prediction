import os
import sys
import numpy as np
import pandas as pd
import joblib
from src.logger import logging
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save an object to a file using joblib

    Args:
        file_path: Path where the object will be saved
        obj: Python object to save
    """
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save object
        joblib.dump(obj, file_path)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load an object from a file using joblib

    Args:
        file_path: Path to the saved object

    Returns:
        Loaded Python object
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load object
        obj = joblib.load(file_path)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        logging.error(f"Error loading object: {e}")
        raise CustomException(e, sys)


def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for required columns and minimum number of rows

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required

    Returns:
        Boolean indicating if the DataFrame is valid
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            logging.warning("DataFrame is empty")
            return False

        # Check if DataFrame has minimum number of rows
        if len(df) < min_rows:
            logging.warning(
                f"DataFrame has {len(df)} rows, minimum required is {min_rows}"
            )
            return False

        # Check for required columns
        if required_columns is not None:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.warning(
                    f"DataFrame is missing required columns: {missing_columns}"
                )
                return False

        return True

    except Exception as e:
        logging.error(f"Error validating DataFrame: {e}")
        raise CustomException(e, sys)


def handle_missing_values(df, numeric_strategy="median", categorical_strategy="mode"):
    """
    Handle missing values in a DataFrame

    Args:
        df: DataFrame with missing values
        numeric_strategy: Strategy for numeric columns ('mean', 'median', 'zero')
        categorical_strategy: Strategy for categorical columns ('mode', 'missing')

    Returns:
        DataFrame with missing values handled
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            logging.warning("DataFrame is empty, cannot handle missing values")
            return df

        # Create a copy of the DataFrame
        result = df.copy()

        # Handle numeric columns
        numeric_cols = result.select_dtypes(include=["int64", "float64"]).columns

        for col in numeric_cols:
            if result[col].isnull().sum() > 0:
                if numeric_strategy == "mean":
                    result[col] = result[col].fillna(result[col].mean())
                elif numeric_strategy == "median":
                    result[col] = result[col].fillna(result[col].median())
                elif numeric_strategy == "zero":
                    result[col] = result[col].fillna(0)

                logging.info(
                    f"Filled missing values in numeric column '{col}' with {numeric_strategy}"
                )

        # Handle categorical columns
        categorical_cols = result.select_dtypes(include=["object", "category"]).columns

        for col in categorical_cols:
            if result[col].isnull().sum() > 0:
                if categorical_strategy == "mode":
                    mode_value = result[col].mode()[0]
                    result[col] = result[col].fillna(mode_value)
                elif categorical_strategy == "missing":
                    result[col] = result[col].fillna("missing")

                logging.info(
                    f"Filled missing values in categorical column '{col}' with {categorical_strategy}"
                )

        return result

    except Exception as e:
        logging.error(f"Error handling missing values: {e}")
        raise CustomException(e, sys)


def handle_outliers(df, columns, method="iqr", threshold=1.5):
    """
    Handle outliers in a DataFrame

    Args:
        df: DataFrame with outliers
        columns: List of columns to check for outliers
        method: Method for detecting outliers ('iqr', 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with outliers handled
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            logging.warning("DataFrame is empty, cannot handle outliers")
            return df

        # Create a copy of the DataFrame
        result = df.copy()

        # Check which columns exist in the DataFrame
        existing_columns = [col for col in columns if col in result.columns]

        # Handle outliers for each column
        for col in existing_columns:
            # Skip non-numeric columns
            if not np.issubdtype(result[col].dtype, np.number):
                continue

            if method == "iqr":
                # Calculate IQR
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1

                # Calculate bounds
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

                # Cap outliers
                result[col] = result[col].clip(lower_bound, upper_bound)

                logging.info(f"Handled outliers in column '{col}' using IQR method")

            elif method == "zscore":
                # Calculate Z-score
                mean = result[col].mean()
                std = result[col].std()

                # Cap outliers
                result[col] = result[col].clip(
                    mean - threshold * std, mean + threshold * std
                )

                logging.info(f"Handled outliers in column '{col}' using Z-score method")

        return result

    except Exception as e:
        logging.error(f"Error handling outliers: {e}")
        raise CustomException(e, sys)
