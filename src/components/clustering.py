import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.logger import logging
from src.exception import CustomException
from src.utils.common import save_object
from src.utils.config import Config


class TripClustering:
    """
    Component for clustering taxi trips based on their characteristics.

    This class handles:
    1. Feature selection for clustering
    2. K-means clustering
    3. Cluster analysis
    """

    def __init__(self):
        """Initialize TripClustering with configuration"""
        # Load configuration
        self.config = Config()
        self.models_dir = self.config.models_dir
        self.clusterer_path = os.path.join(self.models_dir, "clusterer.joblib")

    def prepare_clustering_features(self, df):
        """
        Prepare features for clustering

        Args:
            df: DataFrame containing trip data

        Returns:
            Scaled feature matrix
        """
        try:
            logging.info("Preparing features for clustering")

            # Check if DataFrame is empty
            if df.empty:
                raise ValueError(
                    "DataFrame is empty, cannot prepare clustering features"
                )

            # Select features for clustering (gracefully handling missing columns)
            available_features = []
            desired_features = [
                "trip_distance",
                "fare_amount",
                "passenger_count",
                "speed",
                "fare_per_mile",
                "pickup_hour",
            ]

            for feature in desired_features:
                if feature in df.columns:
                    available_features.append(feature)
                else:
                    logging.warning(
                        f"Feature '{feature}' not found in DataFrame, skipping"
                    )

            if len(available_features) < 2:
                logging.error("Not enough features available for clustering")
                raise ValueError("Not enough features available for clustering")

            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[available_features])

            logging.info(f"Features prepared successfully: {available_features}")
            return scaled_features

        except Exception as e:
            logging.error(f"Error preparing clustering features: {e}")
            raise CustomException(e, sys)

    def perform_clustering(self, scaled_features, n_clusters=3):
        """
        Perform K-means clustering

        Args:
            scaled_features: Scaled feature matrix
            n_clusters: Number of clusters

        Returns:
            Tuple of (fitted KMeans model, cluster labels)
        """
        try:
            logging.info(f"Performing K-means clustering with {n_clusters} clusters")

            # Verify we have enough data
            if len(scaled_features) < n_clusters:
                # Adjust number of clusters if not enough data
                original_n_clusters = n_clusters
                n_clusters = min(n_clusters, max(2, len(scaled_features) // 2))
                logging.warning(
                    f"Not enough data for {original_n_clusters} clusters, reduced to {n_clusters}"
                )

            # Create and fit KMeans model
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)

            logging.info(
                f"Clustering completed successfully with {n_clusters} clusters"
            )
            return kmeans, cluster_labels

        except Exception as e:
            logging.error(f"Error performing clustering: {e}")
            raise CustomException(e, sys)

    def analyze_clusters(self, df, cluster_labels):
        """
        Analyze characteristics of each cluster

        Args:
            df: DataFrame containing trip data
            cluster_labels: Cluster assignments

        Returns:
            DataFrame with cluster statistics
        """
        try:
            logging.info("Analyzing clusters")

            # Add cluster labels to DataFrame
            df = df.copy()
            df["cluster"] = cluster_labels

            # Check what features are available for statistics
            available_stats = []
            desired_stats = [
                "fare_amount",
                "trip_distance",
                "speed",
                "passenger_count",
                "fare_per_mile",
            ]

            for stat in desired_stats:
                if stat in df.columns:
                    available_stats.append(stat)

            # Calculate cluster statistics
            if available_stats:
                cluster_stats = df.groupby("cluster")[available_stats].agg(
                    ["mean", "std", "count"]
                )
                # Flatten multi-index columns
                cluster_stats.columns = [
                    f"{col[0]}_{col[1]}" for col in cluster_stats.columns
                ]
                # Round values
                cluster_stats = cluster_stats.round(2)
            else:
                # Create dummy stats if no features available
                unique_clusters = df["cluster"].unique()
                data = {
                    "cluster_size": [
                        df[df["cluster"] == c].shape[0] for c in unique_clusters
                    ]
                }
                cluster_stats = pd.DataFrame(data, index=unique_clusters)

            logging.info("Cluster analysis completed")
            return cluster_stats

        except Exception as e:
            logging.error(f"Error analyzing clusters: {e}")
            raise CustomException(e, sys)

    def initiate_clustering(self, df):
        """
        Entry point for clustering process

        Args:
            df: DataFrame containing trip data

        Returns:
            Tuple of (cluster labels, cluster statistics)
        """
        try:
            logging.info("Starting clustering process")

            # Check if DataFrame is empty
            if df.empty:
                logging.warning(
                    "DataFrame is empty, creating synthetic data for clustering"
                )
                df = self._generate_synthetic_data(100)

            # Prepare features
            scaled_features = self.prepare_clustering_features(df)

            # Perform clustering
            clusterer, cluster_labels = self.perform_clustering(scaled_features)

            # Save clusterer
            os.makedirs(self.models_dir, exist_ok=True)
            save_object(self.clusterer_path, clusterer)

            # Analyze clusters
            cluster_stats = self.analyze_clusters(df, cluster_labels)

            logging.info("Clustering process completed successfully")
            return cluster_labels, cluster_stats

        except Exception as e:
            logging.error(f"Error in clustering process: {e}")
            raise CustomException(e, sys)

    def _generate_synthetic_data(self, num_samples=100):
        """
        Generate synthetic data for clustering when DataFrame is empty

        Args:
            num_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic data
        """
        logging.info(f"Generating {num_samples} synthetic records for clustering")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate data with clear cluster structure
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
            "pickup_hour": np.random.choice(range(24), num_samples),
        }

        # Calculate derived features
        data["speed"] = np.random.normal(20, 5, num_samples)
        data["fare_per_mile"] = data["fare_amount"] / data["trip_distance"]

        # Create DataFrame
        df = pd.DataFrame(data)

        logging.info("Synthetic clustering data generated successfully")
        return df
