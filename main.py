import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.database import DatabaseConnector
from src.utils.config import Config


def main():
    """
    Main entry point for the application.

    This function:
    1. Creates necessary directories
    2. Sets up logging
    3. Initializes the database
    4. Runs the training pipeline
    """
    try:
        # Initialize configuration
        config = Config()

        # Initialize database
        logging.info("Initializing database")

        with DatabaseConnector(config.db_path) as db:
            # Create table and initialize with sample data
            db.create_table("all_taxi_trips")

        # Run training pipeline
        logging.info("Starting training pipeline")
        pipeline = TrainingPipeline()

        # Set use_synthetic=False to ensure only database data is used
        results = pipeline.run_pipeline(
            sample_size=config.sample_size, use_synthetic=False
        )

        # Print results
        print("\nTraining Results:")
        print("----------------")

        # Clustering results
        print("\nClustering Statistics:")
        for cluster, stats in results["clustering"]["cluster_stats"].items():
            print(f"\nCluster {cluster}:")
            for metric, value in stats.items():
                print(f"  {metric}: {value:.2f}")

        # Classification results
        print("\nClassification Metrics:")
        print(f"Accuracy: {results['classification']['accuracy']:.3f}")
        print("\nClassification Report:")
        for category, metrics in results["classification"][
            "classification_report"
        ].items():
            if category not in ["accuracy", "macro avg", "weighted avg"]:
                print(f"\n{category}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.3f}")

        # Regression results
        print("\nRegression Metrics:")
        for metric, value in results["regression"].items():
            print(f"{metric.upper()}: {value:.3f}")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
