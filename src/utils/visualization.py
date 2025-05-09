import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.logger import logging
from src.exception import CustomException

class Visualizer:
    """
    Utility class for creating visualizations.
    
    This class provides methods for generating various plots and visualizations
    to analyze model performance and data characteristics.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize Visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir if output_dir else os.path.join(os.getcwd(), "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('fivethirtyeight')
    
    def save_fig(self, fig, filename):
        """
        Save a Plotly figure
        
        Args:
            fig: Plotly figure object
            filename: Name of the file to save
        """
        try:
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(filepath)
            logging.info(f"Figure saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving figure: {e}")
            raise CustomException(e, sys)
    
    def plot_fare_distribution(self, df, category_col='fare_category'):
        """
        Plot fare category distribution
        
        Args:
            df: DataFrame with fare categories
            category_col: Column name for fare categories
            
        Returns:
            Plotly figure object
        """
        try:
            # Get category distribution
            cat_dist = df[category_col].value_counts().reset_index()
            cat_dist.columns = ['category', 'count']
            
            # Create pie chart
            fig = px.pie(
                cat_dist, 
                values='count', 
                names='category',
                title='Distribution of Fare Categories',
                color='category',
                color_discrete_map={'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting fare distribution: {e}")
            raise CustomException(e, sys)
    
    def plot_fare_vs_distance(self, df, color_col='fare_category'):
        """
        Plot fare amount vs. trip distance
        
        Args:
            df: DataFrame with trip data
            color_col: Column name for color coding
            
        Returns:
            Plotly figure object
        """
        try:
            # Sample data for better visualization
            sample_df = df.sample(min(500, len(df)), random_state=42)
            
            # Create scatter plot
            fig = px.scatter(
                sample_df,
                x='trip_distance',
                y='fare_amount',
                color=color_col,
                title='Fare Amount vs. Trip Distance',
                labels={'trip_distance': 'Trip Distance (miles)', 'fare_amount': 'Fare Amount ($)'}
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting fare vs distance: {e}")
            raise CustomException(e, sys)
    
    def plot_cluster_profiles(self, cluster_profile):
        """
        Plot cluster profiles
        
        Args:
            cluster_profile: DataFrame with cluster profiles
            
        Returns:
            Plotly figure object
        """
        try:
            # Reshape for plotting
            profile_long = cluster_profile.reset_index().melt(
                id_vars='cluster',
                var_name='feature',
                value_name='value'
            )
            
            # Create bar chart
            fig = px.bar(
                profile_long,
                x='cluster',
                y='value',
                color='feature',
                barmode='group',
                title='Cluster Profiles by Feature'
            )
            
            fig.update_layout(
                xaxis_title='Cluster',
                yaxis_title='Mean Value'
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting cluster profiles: {e}")
            raise CustomException(e, sys)
    
    def plot_clusters_pca(self, df, cluster_features):
        """
        Visualize clusters using PCA
        
        Args:
            df: DataFrame with cluster assignments
            cluster_features: Features used for clustering
            
        Returns:
            Plotly figure object
        """
        try:
            # Select features and scale
            from sklearn.preprocessing import StandardScaler
            X_cluster = df[cluster_features].copy()
            X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
            
            # Apply PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_cluster_scaled)
            
            # Create DataFrame with PCA results
            pca_df = pd.DataFrame({
                'PCA1': X_pca[:, 0],
                'PCA2': X_pca[:, 1],
                'cluster': df['cluster'],
                'fare_category': df['fare_category']
            })
            
            # Create scatter plot
            fig = px.scatter(
                pca_df,
                x='PCA1',
                y='PCA2',
                color='cluster',
                title='Cluster Visualization (PCA)',
                labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
                hover_data=['fare_category']
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting clusters with PCA: {e}")
            raise CustomException(e, sys)
    
    def plot_fare_category_by_cluster(self, df):
        """
        Plot fare category distribution by cluster
        
        Args:
            df: DataFrame with clusters and fare categories
            
        Returns:
            Plotly figure object
        """
        try:
            # Create crosstab
            crosstab = pd.crosstab(
                df['cluster'], 
                df['fare_category'],
                normalize='index'
            ) * 100
            
            # Reshape for plotting
            crosstab_long = crosstab.reset_index().melt(
                id_vars='cluster',
                var_name='fare_category',
                value_name='percentage'
            )
            
            # Create bar chart
            fig = px.bar(
                crosstab_long,
                x='cluster',
                y='percentage',
                color='fare_category',
                title='Fare Category Distribution by Cluster',
                labels={'percentage': 'Percentage (%)', 'cluster': 'Cluster'},
                color_discrete_map={'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting fare category by cluster: {e}")
            raise CustomException(e, sys)
    
    def plot_classification_performance(self, results_df):
        """
        Plot classification performance comparison
        
        Args:
            results_df: DataFrame with classifier results
            
        Returns:
            Plotly figure object
        """
        try:
            # Create bar chart
            fig = px.bar(
                results_df,
                x='Classifier',
                y=['Accuracy', 'F1 Score'],
                title='Classification Performance by Algorithm',
                barmode='group'
            )
            
            fig.update_layout(
                yaxis_title='Score',
                xaxis_title='Classifier'
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting classification performance: {e}")
            raise CustomException(e, sys)
    
    def plot_confusion_matrix(self, cm, class_names):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            
        Returns:
            Plotly figure object
        """
        try:
            # Create heatmap
            fig = px.imshow(
                cm,
                x=class_names,
                y=class_names,
                text_auto=True,
                title='Confusion Matrix',
                labels=dict(x="Predicted", y="True", color="Count")
            )
            
            fig.update_layout(
                xaxis_title="Predicted Class",
                yaxis_title="True Class"
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")
            raise CustomException(e, sys)
    
    def plot_feature_importance(self, feature_importance):
        """
        Plot feature importance
        
        Args:
            feature_importance: DataFrame with feature names and importances
            
        Returns:
            Plotly figure object
        """
        try:
            # Get top features
            top_features = feature_importance.head(8)
            
            # Create horizontal bar chart
            fig = px.bar(
                top_features,
                x='Importance',
                y='Feature',
                title='Feature Importance',
                orientation='h'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting feature importance: {e}")
            raise CustomException(e, sys)
    
    def plot_regression_performance(self, results_df):
        """
        Plot regression performance comparison
        
        Args:
            results_df: DataFrame with regressor results
            
        Returns:
            Plotly figure object
        """
        try:
            # Create bar chart
            fig = px.bar(
                results_df,
                x='Regressor',
                y=['MAE', 'RMSE'],
                title='Regression Performance by Algorithm',
                barmode='group'
            )
            
            fig.update_layout(
                yaxis_title='Error ($)',
                xaxis_title='Regressor'
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting regression performance: {e}")
            raise CustomException(e, sys)
    
    def plot_actual_vs_predicted(self, df, category_col='predicted_fare_category'):
        """
        Plot actual vs. predicted fares
        
        Args:
            df: DataFrame with actual and predicted fares
            category_col: Column name for color coding
            
        Returns:
            Plotly figure object
        """
        try:
            # Sample data for visualization
            sample_df = df.sample(min(1000, len(df)), random_state=42)
            
            # Create scatter plot
            fig = px.scatter(
                sample_df,
                x='fare_amount',
                y='predicted_fare',
                color=category_col,
                title='Actual vs. Predicted Fare Amounts',
                labels={'fare_amount': 'Actual Fare ($)', 'predicted_fare': 'Predicted Fare ($)'},
                color_discrete_map={'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}
            )
            
            # Add perfect prediction line
            fig.add_trace(
                go.Scatter(
                    x=[sample_df['fare_amount'].min(), sample_df['fare_amount'].max()],
                    y=[sample_df['fare_amount'].min(), sample_df['fare_amount'].max()],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name='Perfect Prediction'
                )
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting actual vs predicted: {e}")
            raise CustomException(e, sys)
    
    def plot_error_distribution(self, df, category_col='predicted_fare_category'):
        """
        Plot prediction error distribution
        
        Args:
            df: DataFrame with predictions and errors
            category_col: Column name for color coding
            
        Returns:
            Plotly figure object
        """
        try:
            # Calculate error if not already present
            if 'prediction_error' not in df.columns:
                df_with_error = df.copy()
                df_with_error['prediction_error'] = df_with_error['predicted_fare'] - df_with_error['fare_amount']
            else:
                df_with_error = df
            
            # Create histogram
            fig = px.histogram(
                df_with_error,
                x='prediction_error',
                color=category_col,
                title='Prediction Error Distribution',
                labels={'prediction_error': 'Prediction Error ($)'},
                color_discrete_map={'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'},
                nbins=50
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting error distribution: {e}")
            raise CustomException(e, sys)
    
    def plot_error_by_feature(self, df, feature_col, feature_name=None):
        """
        Plot error by feature category
        
        Args:
            df: DataFrame with predictions and errors
            feature_col: Column name for feature categories
            feature_name: Display name for the feature
            
        Returns:
            Plotly figure object
        """
        try:
            # Calculate absolute error if not present
            if 'absolute_error' not in df.columns:
                df_with_error = df.copy()
                df_with_error['absolute_error'] = abs(df_with_error['predicted_fare'] - df_with_error['fare_amount'])
            else:
                df_with_error = df
            
            # Group by feature and calculate mean error
            error_by_feature = df_with_error.groupby(feature_col)['absolute_error'].mean().reset_index()
            
            # Set display name
            if feature_name is None:
                feature_name = feature_col.replace('_', ' ').title()
            
            # Create bar chart
            fig = px.bar(
                error_by_feature,
                x=feature_col,
                y='absolute_error',
                title=f'Mean Absolute Error by {feature_name}',
                color=feature_col
            )
            
            fig.update_layout(
                xaxis_title=feature_name,
                yaxis_title='Mean Absolute Error ($)'
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting error by feature: {e}")
            raise CustomException(e, sys)
    
    def plot_model_comparison(self, comparison_df):
        """
        Plot comparison between models
        
        Args:
            comparison_df: DataFrame with model comparisons
            
        Returns:
            Plotly figure object
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Error Metrics by Model", "R² Score by Model")
            )
            
            # Add MAE bars
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df['MAE'],
                    name='MAE',
                    marker_color='blue'
                ),
                row=1, col=1
            )
            
            # Add R2 bars
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df['R2'],
                    name='R²',
                    marker_color='green'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text='Model Comparison',
                showlegend=False,
                height=400
            )
            
            fig.update_yaxes(title_text='Mean Absolute Error ($)', row=1, col=1)
            fig.update_yaxes(title_text='R² Score', row=1, col=2)
            
            return fig
        except Exception as e:
            logging.error(f"Error plotting model comparison: {e}")
            raise CustomException(e, sys)
    
    def create_dashboard(self, results, prefix='dashboard'):
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Args:
            results: Dictionary with pipeline results
            prefix: Prefix for saved files
            
        Returns:
            List of saved file paths
        """
        try:
            logging.info("Creating visualization dashboard")
            saved_files = []
            
            # Extract data from results
            final_data = results['final_data']
            cluster_analysis = results['cluster_analysis']
            classification_results = results['classification_results']
            regression_results = results['regression_results']
            
            # 1. Fare distribution
            fig1 = self.plot_fare_distribution(final_data)
            filename1 = f"{prefix}_fare_distribution.html"
            self.save_fig(fig1, filename1)
            saved_files.append(filename1)
            
            # 2. Fare vs distance
            fig2 = self.plot_fare_vs_distance(final_data)
            filename2 = f"{prefix}_fare_vs_distance.html"
            self.save_fig(fig2, filename2)
            saved_files.append(filename2)
            
            # 3. Cluster profiles
            fig3 = self.plot_cluster_profiles(cluster_analysis['cluster_profile'])
            filename3 = f"{prefix}_cluster_profiles.html"
            self.save_fig(fig3, filename3)
            saved_files.append(filename3)
            
            # 4. Clusters PCA
            cluster_features = cluster_analysis['cluster_profile'].columns.tolist()
            fig4 = self.plot_clusters_pca(final_data, cluster_features)
            filename4 = f"{prefix}_clusters_pca.html"
            self.save_fig(fig4, filename4)
            saved_files.append(filename4)
            
            # 5. Fare category by cluster
            fig5 = self.plot_fare_category_by_cluster(final_data)
            filename5 = f"{prefix}_fare_category_by_cluster.html"
            self.save_fig(fig5, filename5)
            saved_files.append(filename5)
            
            # 6. Feature importance
            if classification_results.get('feature_importance') is not None:
                fig6 = self.plot_feature_importance(classification_results['feature_importance'])
                filename6 = f"{prefix}_feature_importance.html"
                self.save_fig(fig6, filename6)
                saved_files.append(filename6)
            
            # 7. Actual vs predicted
            fig7 = self.plot_actual_vs_predicted(final_data)
            filename7 = f"{prefix}_actual_vs_predicted.html"
            self.save_fig(fig7, filename7)
            saved_files.append(filename7)
            
            # 8. Error distribution
            fig8 = self.plot_error_distribution(final_data)
            filename8 = f"{prefix}_error_distribution.html"
            self.save_fig(fig8, filename8)
            saved_files.append(filename8)
            
            # 9. Error by distance category
            fig9 = self.plot_error_by_feature(final_data, 'distance_category', 'Distance Category')
            filename9 = f"{prefix}_error_by_distance.html"
            self.save_fig(fig9, filename9)
            saved_files.append(filename9)
            
            # 10. Model comparison
            # Create comparison dataframe
            if regression_results.get('model_comparison') is not None:
                comparison = regression_results['model_comparison']
                comparison_df = pd.DataFrame({
                    'Model': ['Specialized Models', 'Baseline Model'],
                    'MAE': [comparison['specialized_metrics']['MAE'], comparison['baseline_metrics']['MAE']],
                    'RMSE': [comparison['specialized_metrics']['RMSE'], comparison['baseline_metrics']['RMSE']],
                    'R2': [comparison['specialized_metrics']['R2'], comparison['baseline_metrics']['R2']]
                })
                
                fig10 = self.plot_model_comparison(comparison_df)
                filename10 = f"{prefix}_model_comparison.html"
                self.save_fig(fig10, filename10)
                saved_files.append(filename10)
            
            logging.info(f"Dashboard created with {len(saved_files)} visualizations")
            return saved_files
            
        except Exception as e:
            logging.error(f"Error creating dashboard: {e}")
            raise CustomException(e, sys)