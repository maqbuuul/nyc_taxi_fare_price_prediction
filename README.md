# NYC Taxi Fare Prediction: Comprehensive Project Documentation

## Problem Statement

New York City's taxi system processes millions of trips monthly, but both drivers and passengers face challenges with fare predictability. Drivers struggle to optimize routes and estimate earnings, while passengers encounter uncertainty about costs before booking. This project addresses these challenges by developing a robust, multi-stage machine learning pipeline to accurately predict taxi fares based on trip characteristics, empowering both drivers and passengers with reliable fare estimates while providing transportation planners with actionable insights into urban mobility patterns.

![NYC Taxi](config/1OxIdFjt7v3wCErqGfSwD6w.jpg)

## Project Overview

This comprehensive data science project analyzes New York City taxi trip data to develop an accurate fare prediction system. Through extensive exploratory data analysis and advanced modeling techniques, I uncovered key patterns, relationships, and insights that inform a sophisticated multi-stage prediction approach while revealing the underlying dynamics of NYC's transportation ecosystem.

## Features & Capabilities

- **Multi-stage prediction pipeline**:
  1. Trip clustering to identify similar trip patterns
  2. Fare category classification (low, medium, high)
  3. Specialized regression models for each fare category
- **Robust error handling and logging**: Comprehensive exception handling and logging system
- **Advanced visualization utilities**: Tools for analyzing model performance and data patterns
- **Hyperparameter optimization**: Grid search and cross-validation for optimal model tuning
- **Production-ready architecture**: Modular design ready for deployment

## Modular Implementation Approach

The project follows software engineering best practices with a component-based architecture:

- **Data Pipeline Module**: Handles database connections, data extraction, and transformations
- **Feature Engineering Module**: Creates derived features, handles categorical encoding, and standardization
- **Modeling Pipeline Module**: Implements the multi-stage ML approach (clustering, classification, regression)
- **Evaluation Module**: Calculates metrics, generates visualizations, and compares model performance
- **Persistence Module**: Handles model serialization, versioning, and loading for production use

This modular approach enhances maintainability, facilitates collaboration, and enables easier deployment to production environments.

## Project Structure

```
nyc_taxi_fare_prediction/
│
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
├── setup.py                      # Package setup script
│
├── src/                          # Source code
│   ├── __init__.py               # Package initialization
│   ├── exception.py              # Custom exception handling
│   ├── logger.py                 # Logging configuration
│   │
│   ├── components/               # Modular components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py     # Data loading and preparation
│   │   ├── data_transformation.py # Feature engineering and preprocessing
│   │   ├── clustering.py         # Trip clustering logic
│   │   ├── classification.py     # Fare category classification
│   │   └── regression.py         # Specialized regression models
│   │
│   ├── pipelines/                # Workflow pipelines
│   │   ├── __init__.py
│   │   ├── training_pipeline.py  # End-to-end training workflow
│   │   └── prediction_pipeline.py # Prediction service
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── common.py             # Shared utilities
│       ├── database.py           # Database connection utilities
│       ├── evaluation.py         # Model evaluation utilities
│       └── visualization.py      # Plotting and visualization utilities
│
├── config/                       # Configuration files
│   └── config.yaml               # Project configuration
│
├── models/                       # Saved model files
│
├── notebooks/                    # Jupyter notebooks for exploration
│
└── logs/                         # Log files
```

## Installation & Usage

### Installation

1. Clone the repository:
```bash
git clone https://github.com/abdiwahidali/nyc_taxi_fare_prediction.git
cd nyc_taxi_fare_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Models

To train the full model pipeline:

```python
from src.pipelines.training_pipeline import TrainingPipeline

# Create and run the training pipeline
pipeline = TrainingPipeline()
results = pipeline.run_pipeline(sample_size=10000)
```

### Making Predictions

To make predictions on new data:

```python
from src.pipelines.prediction_pipeline import PredictionAPI

# Create the prediction API
api = PredictionAPI()

# Predict for a single trip
trip_data = {
    'trip_distance': 5.2,
    'passenger_count': 2,
    'pickup_hour': 18,
    'payment_type': 1  # Credit card
}

result = api.predict_single_trip(trip_data)
print(f"Predicted fare: ${result['predicted_fare']:.2f}")
print(f"Fare category: {result['predicted_fare_category']}")
```

## Data & Methodology

The analysis leverages a substantial dataset of NYC taxi trips stored in a DuckDB database. I implemented a robust data pipeline using Python, Pandas, and SQL to extract, clean, and prepare the data for analysis. The methodology followed these key steps:

1. **Data Acquisition**: Connected to the database and extracted samples of taxi trips for analysis
2. **Data Cleaning**: Removed outliers, handled missing values, and standardized data formats
3. **Feature Engineering**: Created derived features including cost per mile, distance categories, time-of-day metrics, and fare classifications
4. **Exploratory Analysis**: Developed interactive visualizations using Plotly, Seaborn, and Folium
5. **Model Development**: Implemented a multi-stage pipeline with clustering, classification, and specialized regression
6. **Hyperparameter Tuning**: Optimized model parameters through grid search and cross-validation
7. **Evaluation**: Assessed model performance through various metrics and comparative analysis

## Component Details

### 1. Data Ingestion (`data_ingestion.py`)

This component handles loading data from the DuckDB database and performing basic cleaning operations:

- Connects to the database using configurable connection parameters
- Extracts a configurable sample size of taxi trip data
- Implements initial data validation and cleaning steps
- Handles exceptions with proper error logging
- Saves processed data for downstream components

### 2. Data Transformation (`data_transformation.py`)

Handles feature engineering, creating new features from raw data, and preparing data for modeling:

- Creates derived features like cost per mile, distance categories, etc.
- Implements preprocessing pipelines for numerical and categorical features
- Handles categorical encoding and numerical scaling
- Creates feature transformers that can be serialized for a prediction pipeline
- Includes data validation and outlier handling

### 3. Trip Clustering (`clustering.py`)

Implements K-means clustering to identify distinct patterns in taxi trips:

- Optimizes cluster count using the silhouette score and the Davies-Bouldin index
- Extracts cluster profiles and characteristics
- Visualizes cluster distributions and relationships
- Creates and serializes the clustering model for prediction
- Adds cluster assignments as features for downstream models

### 4. Fare Category Classification (`classification.py`)

Trains classifiers to predict fare categories (low, medium, high):

- Implements multiple classification algorithms (Random Forest, XGBoost, etc.)
- Compares algorithm performance with cross-validation
- Performs feature importance analysis
- Creates confusion matrix and classification performance visualizations
- Serializes the best classifier for the prediction pipeline

### 5. Specialized Regression (`regression.py`)

Trains dedicated regression models for each fare category:

- Implements specialized models for low, medium, and high fare categories
- Compares algorithm performance within each category
- Optimizes hyperparameters for best-performing models
- Evaluates models on various metrics (MAE, RMSE, R²)
- Serializes models for deployment in the prediction pipeline

## Key EDA Findings

### Trip Distance & Fare Relationship

Trip distance emerged as the strongest predictor of fare amount, with a clear positive correlation. However, the relationship is not linear, particularly for short or long trips.

The analysis revealed that the cost per mile decreases significantly as trip distance increases, suggesting a non-linear fare structure that includes both fixed and variable components:

- Very short trips (<1 mile) average $9.34 per mile
- Medium trips (2-3 miles) average $5.87 per mile
- Long trips (10+ miles) average $3.21 per mile

This insight became crucial for developing specialized regression models that accurately handle trips of varying distances.

### Payment Patterns & Tipping Behavior

The analysis uncovered distinct patterns in payment methods and tipping behavior:

- Credit card payments make up 63.7% of all trips
- Cash payments account for 35.2% of trips
- Credit card users tip an average of 15.8% of the fare amount
- Trips paid by credit card have slightly higher average fare amounts

These insights suggest that payment type serves as a valuable predictor and informed the decision to include payment features in the modeling pipeline.

### Passenger Count Influence

Surprisingly, passenger count showed minimal effect on fare amounts, suggesting it may not be a strong predictor for modeling:

- Solo passengers (1 person) represent 59.4% of all trips
- Couples (2 people) account for 24.7% of trips
- Larger groups show only marginally higher average fare amounts

This finding challenges common assumptions about group size impacting fare pricing. The modeling results confirmed this, with passenger count contributing minimally to prediction accuracy.

### Geographic Distribution & Route Analysis

The geographic analysis revealed significant spatial patterns in pickup and drop-off locations:

- High concentration of pickups in Midtown and Lower Manhattan
- Notable dropoff clusters at major transportation hubs (airports, train stations)
- Specific location pairs showed consistently higher fare amounts, likely due to fixed pricing for airport routes

A particularly interesting finding was the identification of "route inequality" - certain routes consistently showed higher cost per mile compared to others of similar distance, potentially indicating opportunities for route optimization.

### Rate Code Impact

Different rate codes have distinct fare structures, with significant variations in pricing:

- Standard rate (RateCodeID 1): Used for 87.3% of trips
- JFK Airport fixed fare (RateCodeID 2): Shows consistently higher fares regardless of actual distance
- Newark Airport (RateCodeID 3): Exhibits the highest average cost per mile

These specialized rate codes represent important edge cases that require separate handling in a prediction model. The clustering approach effectively identified these distinct trip types.

## Modeling Insights

### Clustering Analysis

The K-means clustering algorithm identified optimal trip segments based on distance, fare amount, cost per mile, and other features:

- **Cluster 0**: Short, expensive trips (high cost per mile, low distance)
- **Cluster 1**: Medium-distance trips with standard pricing
- **Cluster 2**: Long-distance trips with discounted per-mile rates
- **Cluster 3**: Airport and special fare trips with fixed pricing structures

The silhouette score analysis indicated that 4 clusters provided the optimal segmentation, with diminishing returns for additional clusters. These clusters served as valuable features for downstream classification and regression models.

### Fare Category Classification

A comparative analysis of classification algorithms revealed:

- **Random Forest** achieved the highest F1 score (0.89) for classifying trips into low, medium, and high fare categories
- **XGBoost** showed comparable performance (F1 = 0.87) with faster training times
- **Gradient Boosting** performed well but required more extensive tuning

Feature importance analysis highlighted that trip distance, cluster assignment, and cost per mile were the most predictive features for fare category classification.

### Specialized Regression Models

The multi-stage approach with specialized regression models for each fare category significantly outperformed baseline models:

- **Overall Performance**: 18.7% reduction in Mean Absolute Error compared to a single model approach
- **Low Fare Category**: Random Forest performed best (MAE = $1.24)
- **Medium Fare Category**: Gradient Boosting excelled (MAE = $2.18)
- **High Fare Category**: XGBoost showed superior performance (MAE = $4.51)

Hyperparameter tuning further improved performance, particularly for the medium fare category, reducing MAE by an additional 8.3%.

### Model Accuracy by Error Tolerance

The specialized model approach demonstrated superior real-world applicability:

- **Within $1**: 62.8% of predictions (vs. 54.1% for baseline)
- **Within $2**: 83.7% of predictions (vs. 76.3% for baseline)
- **Within $5**: 96.2% of predictions (vs. 91.8% for baseline)

These results indicate that the multi-stage pipeline provides fare estimates within an acceptable margin for the vast majority of trips, making it suitable for deployment in customer-facing applications.

## Technical Implementation

The project utilizes a powerful technical stack:

- **Python** for data manipulation, analysis, and modeling
- **Pandas & NumPy** for data structures and calculations
- **Scikit-learn** for modeling pipeline, preprocessing, and evaluation
- **XGBoost** for gradient boosted tree models
- **Plotly, Seaborn & Matplotlib** for interactive visualizations
- **Folium** for geographic mapping
- **DuckDB** for efficient SQL queries on large datasets
- **Pickle** for model serialization and persistence

The implementation emphasizes performance optimization, scalable data handling techniques, and modular code organization suitable for enterprise deployment.

## Recommendations

Based on the comprehensive analysis and modeling results, I recommend the following actions:

### For Taxi Service Providers

1. **Implement the multi-stage prediction pipeline** for accurate fare estimates in booking applications
2. **Develop specialized pricing models** for different trip types identified by clustering
3. **Optimize driver placement** based on geographic demand patterns
4. **Review pricing structures** for very short trips, which show disproportionately high cost per mile
5. **Enhance transparency** around fixed-fare routes (airports, special destinations)

### For Transportation Planners

1. **Address route inequality** by investigating high-cost corridors
2. **Optimize public transportation** along high-demand taxi routes
3. **Review fare regulations** for consistency and fairness across boroughs
4. **Consider incentives** for shared rides to improve vehicle utilization

### For Model Enhancement

1. **Integrate external data sources** such as weather conditions, event calendars, and traffic patterns
2. **Implement real-time model updates** to adapt to changing conditions
3. **Develop a monitoring system** to detect drift in model performance
4. **Create a robust API service** for integration with mobile applications
5. **Explore deep learning approaches** for capturing complex spatiotemporal patterns

## Business Impact

The insights and models from this project offer significant value:

- **For passengers**: More accurate fare estimation before booking, with 96.2% of predictions within $5 of the actual fare
- **For drivers**: Optimized route selection and better earning potential through predictable fare estimation
- **For service providers**: Enhanced customer satisfaction through fare transparency and reduced disputes
- **For transportation planners**: Data-driven insights into urban mobility patterns and fare equity

The multi-stage prediction approach, informed by thorough EDA and sophisticated modeling techniques, enables more transparent, efficient, and equitable transportation services.

## Future Work

Building on this project, future work will include:

- **Deploying models as microservices** with API endpoints for real-time prediction
- **Implementing a continuous learning system** with feedback loops from actual fares
- **Developing a demand forecasting module** to complement fare prediction
- **Creating interactive dashboards** for transportation planners and policy makers
- **Exploring reinforcement learning** for dynamic pricing optimization

## Conclusion

This NYC Taxi Fare Prediction project demonstrates my ability to extract meaningful insights from complex urban mobility data and develop sophisticated machine learning solutions. Through methodical data exploration, statistical analysis, and advanced modeling techniques, I uncovered key patterns that informed a multi-stage prediction approach with significantly improved accuracy over baseline methods.

The modular implementation approach ensures code maintainability and scalability, while the comprehensive evaluation framework provides confidence in the model's real-world applicability. The insights gained extend beyond fare prediction to reveal deeper patterns in urban transportation dynamics, offering value to multiple stakeholders in the urban mobility ecosystem.

---

*Technologies: Python, Pandas, NumPy, Scikit-learn, XGBoost, Plotly, Seaborn, Matplotlib, Folium, DuckDB, SQL, Pickle*

**Author:** Abdiwahid Ali

*Contact: [LinkedIn Profile](https://www.linkedin.com/in/https://www.linkedin.com/in/maqbuuul/)*

*Dataset: [download dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)*
