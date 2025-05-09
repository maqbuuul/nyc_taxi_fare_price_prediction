import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import duckdb
from src.logger import logging
from src.exception import CustomException


class DatabaseConnector:
    """
    Class for connecting to the DuckDB database and performing operations.

    This class handles:
    1. Connection to DuckDB
    2. Creating tables
    3. Loading data
    4. Querying data
    """

    def __init__(self, db_path):
        """
        Initialize the DatabaseConnector

        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path
        self.connection = None
        self.cursor = None

    def __enter__(self):
        """Context manager entry point"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            # Connect to DuckDB
            self.connection = duckdb.connect(self.db_path)
            self.cursor = self.connection.cursor()
            logging.info(f"Connected to database at {self.db_path}")
            return self
        except Exception as e:
            logging.error(f"Error connecting to database: {e}")
            raise CustomException(e, sys)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point"""
        try:
            if self.connection:
                self.connection.close()
                logging.info("Database connection closed")
        except Exception as e:
            logging.error(f"Error closing database connection: {e}")

    def create_table(self, table_name):
        """
        Create a table for storing taxi trip data

        Args:
            table_name: Name of the table to create
        """
        try:
            logging.info(f"Creating table {table_name}")

            # First, drop the table if it exists to avoid schema mismatches
            drop_stmt = f"DROP TABLE IF EXISTS {table_name}"
            self.cursor.execute(drop_stmt)
            logging.info(f"Dropped existing table {table_name} if it existed")

            # Create table schema
            create_stmt = f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                VendorID INTEGER,
                tpep_pickup_datetime TIMESTAMP,
                tpep_dropoff_datetime TIMESTAMP,
                passenger_count INTEGER,
                trip_distance FLOAT,
                RatecodeID INTEGER,
                store_and_fwd_flag VARCHAR,
                PULocationID INTEGER,
                DOLocationID INTEGER,
                payment_type INTEGER,
                fare_amount FLOAT,
                extra FLOAT,
                mta_tax FLOAT,
                tip_amount FLOAT,
                tolls_amount FLOAT,
                improvement_surcharge FLOAT,
                total_amount FLOAT,
                trip_time_in_secs INTEGER
            )
            """

            self.cursor.execute(create_stmt)
            logging.info(f"Table {table_name} created successfully")

            # Generate sample data
            sample_data = self._generate_sample_data()

            # Insert sample data
            self._insert_sample_data(table_name, sample_data)

        except Exception as e:
            logging.error(f"Error creating table: {e}")
            raise CustomException(e, sys)

    def _generate_sample_data(self, num_samples=1000):
        """
        Generate sample taxi trip data

        Args:
            num_samples: Number of samples to generate

        Returns:
            DataFrame with sample data
        """
        try:
            logging.info(f"Generating {num_samples} sample records")

            # Set random seed for reproducibility
            np.random.seed(42)

            # Generate data
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            data = {
                "id": list(range(1, num_samples + 1)),
                "VendorID": np.random.choice([1, 2], num_samples).tolist(),
                "tpep_pickup_datetime": [current_time] * num_samples,
                "tpep_dropoff_datetime": [current_time] * num_samples,
                "passenger_count": np.random.choice(range(1, 7), num_samples).tolist(),
                "trip_distance": np.random.uniform(0.5, 20, num_samples).tolist(),
                "RatecodeID": np.random.choice(range(1, 7), num_samples).tolist(),
                "store_and_fwd_flag": np.random.choice(
                    ["Y", "N"], num_samples
                ).tolist(),
                "PULocationID": np.random.randint(1, 265, num_samples).tolist(),
                "DOLocationID": np.random.randint(1, 265, num_samples).tolist(),
                "payment_type": np.random.choice([1, 2, 3, 4], num_samples).tolist(),
                "fare_amount": np.random.uniform(2.5, 100, num_samples).tolist(),
                "extra": np.random.uniform(0, 5, num_samples).tolist(),
                "mta_tax": np.random.uniform(0, 1, num_samples).tolist(),
                "tip_amount": np.random.uniform(0, 20, num_samples).tolist(),
                "tolls_amount": np.random.uniform(0, 10, num_samples).tolist(),
                "improvement_surcharge": np.random.uniform(0, 1, num_samples).tolist(),
                "total_amount": np.random.uniform(5, 150, num_samples).tolist(),
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

            logging.info("Sample data generated successfully")
            return df

        except Exception as e:
            logging.error(f"Error generating sample data: {e}")
            raise CustomException(e, sys)

    def _insert_sample_data(self, table_name, df):
        """
        Insert sample data into the database

        Args:
            table_name: Name of the table
            df: DataFrame with sample data
        """
        try:
            # Use direct SQL batch insert
            values_list = []

            # Prepare only 100 rows for faster execution
            for i, row in df.iloc[:100000].iterrows():
                values = (
                    int(row["id"]),
                    int(row["VendorID"]),
                    str(row["tpep_pickup_datetime"]),
                    str(row["tpep_dropoff_datetime"]),
                    int(row["passenger_count"]),
                    float(row["trip_distance"]),
                    int(row["RatecodeID"]),
                    str(row["store_and_fwd_flag"]),
                    int(row["PULocationID"]),
                    int(row["DOLocationID"]),
                    int(row["payment_type"]),
                    float(row["fare_amount"]),
                    float(row["extra"]),
                    float(row["mta_tax"]),
                    float(row["tip_amount"]),
                    float(row["tolls_amount"]),
                    float(row["improvement_surcharge"]),
                    float(row["total_amount"]),
                    int(row["trip_time_in_secs"]),
                )
                values_list.append(values)

            # Use DuckDB's execute_many for batch insert (more efficient)
            insert_sql = f"""
            INSERT INTO {table_name} (
                id, VendorID, tpep_pickup_datetime, tpep_dropoff_datetime, 
                passenger_count, trip_distance, RatecodeID, store_and_fwd_flag, 
                PULocationID, DOLocationID, payment_type, fare_amount, extra, 
                mta_tax, tip_amount, tolls_amount, improvement_surcharge, 
                total_amount, trip_time_in_secs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            # Execute batch insert
            for values in values_list:
                self.cursor.execute(insert_sql, values)

            # Commit the changes
            self.connection.commit()

            logging.info(f"Data saved to table {table_name}")

        except Exception as e:
            logging.error(f"Error saving data: {e}")
            raise CustomException(e, sys)

    def get_sample(self, table_name, sample_size):
        """
        Get a sample of records from the database

        Args:
            table_name: Name of the table
            sample_size: Number of records to sample

        Returns:
            DataFrame with sampled records
        """
        try:
            # Check if table exists
            check_query = f"""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='{table_name}'
            """
            result = self.cursor.execute(check_query).fetchall()

            if not result:
                logging.warning(
                    f"Table {table_name} doesn't exist, returning empty DataFrame"
                )
                return pd.DataFrame()

            # Check if table has data
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            count = self.cursor.execute(count_query).fetchone()[0]

            if count == 0:
                logging.warning(
                    f"Table {table_name} is empty, returning empty DataFrame"
                )
                return pd.DataFrame()

            # Query database
            query = f"""
            SELECT * FROM {table_name} 
            ORDER BY RANDOM() 
            LIMIT {min(sample_size, count)}
            """

            # Execute query and fetch results
            result = self.cursor.execute(query).fetch_df()

            logging.info(
                f"Successfully retrieved {len(result)} records from {table_name}"
            )
            return result

        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise CustomException(e, sys)
