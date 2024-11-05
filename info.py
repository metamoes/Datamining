import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Union
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrafficAnalyzer:
    """
    Analyzes trained traffic prediction models and generates visualizations.
    """

    def __init__(self, model_path: str, data_dir: str, sequence_length: int = 12):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        self.data_dir = Path(data_dir)

        # Load model checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        logger.info(f"Loaded model from epoch {self.checkpoint['epoch']}")

        # Initialize metrics storage
        self.metrics = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': [],
            'predictions': [],
            'actual_values': [],
            'timestamps': []
        }

        # Define feature columns (matching the training script)
        self._initialize_feature_columns()

    def _initialize_feature_columns(self):
        """Initialize feature columns matching the training script"""
        self.temporal_features = [
            'Hour', 'Day_of_Week', 'Is_Weekend', 'Month', 'Is_Peak_Hour_Normalized'
        ]

        self.road_features = [
            'Station_Length_Normalized', 'Active_Lanes_Normalized',
            'Direction_S_Normalized', 'Direction_E_Normalized', 'Direction_W_Normalized',
            'Lane_Type_FR_Normalized', 'Lane_Type_ML_Normalized', 'Lane_Type_OR_Normalized'
        ]

        self.traffic_features = [
            'Total_Flow_Normalized', 'Avg_Occupancy_Normalized', 'Avg_Speed_Normalized'
        ]

        self.lane_features = []
        for lane in range(1, 5):
            self.lane_features.extend([
                f'Lane_{lane}_Flow_Normalized',
                f'Lane_{lane}_Avg_Occ_Normalized',
                f'Lane_{lane}_Avg_Speed_Normalized',
                f'Lane_{lane}_Efficiency_Normalized'
            ])

        self.feature_cols = (
                self.temporal_features +
                self.road_features +
                self.traffic_features +
                self.lane_features
        )

    def prepare_validation_data(self):
        """Prepare validation dataset from the last 20% of data with NaN handling"""
        all_files = sorted(list(self.data_dir.glob('**/*.csv')))
        total_files = len(all_files)
        val_size = int(0.2 * total_files)
        val_files = all_files[-val_size:]

        sequences = []
        targets = []
        timestamps = []

        for file in tqdm(val_files, desc="Loading validation data"):
            try:
                df = pd.read_csv(file)

                # Check for required columns
                required_cols = self.feature_cols + ['Timestamp', 'Total_Flow_Normalized']
                missing_cols = set(required_cols) - set(df.columns)
                if missing_cols:
                    logger.warning(f"Missing columns in {file}: {missing_cols}")
                    continue

                # Convert timestamp
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])

                # Handle NaN values
                df[self.feature_cols] = df[self.feature_cols].ffill().bfill()
                df['Total_Flow_Normalized'] = df['Total_Flow_Normalized'].ffill().bfill()

                # Skip if still have NaN values
                if df[self.feature_cols + ['Total_Flow_Normalized']].isna().any().any():
                    logger.warning(f"File {file} still contains NaN values after filling")
                    continue

                # Process features
                features = df[self.feature_cols].values
                target_values = df['Total_Flow_Normalized'].values

                # Create sequences
                for i in range(0, len(features) - self.sequence_length):
                    seq = features[i:i + self.sequence_length]
                    target = target_values[i + self.sequence_length]

                    # Only add sequence if it contains no NaN values
                    if not np.isnan(seq).any() and not np.isnan(target):
                        sequences.append(seq)
                        targets.append(target)
                        timestamps.append(df['Timestamp'].iloc[i + self.sequence_length])

            except Exception as e:
                logger.error(f"Error processing file {file}: {str(e)}")
                continue

        if not sequences:
            logger.error("No valid sequences found in the data")
            return torch.FloatTensor(), torch.FloatTensor(), []

        logger.info(f"Successfully prepared {len(sequences)} sequences for validation")
        return (
            torch.FloatTensor(np.array(sequences)),
            torch.FloatTensor(np.array(targets)),
            timestamps
        )

    def evaluate_model(self):
        """Evaluate model performance on validation data with NaN handling"""
        logger.info("Starting model evaluation...")

        # Get validation data
        val_sequences, val_targets, timestamps = self.prepare_validation_data()

        if len(val_sequences) == 0:
            logger.error("No valid validation data available")
            return None

        # Initialize model
        model = TrafficLSTM(
            input_size=val_sequences.shape[2],
            hidden_size=128,
            num_layers=2,
            sequence_length=self.sequence_length
        ).to(self.device)

        # Load model weights
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()

        predictions = []
        valid_targets = []
        valid_timestamps = []

        with torch.no_grad():
            for i in tqdm(range(0, len(val_sequences), 32), desc="Generating predictions"):
                batch_sequences = val_sequences[i:i + 32].to(self.device)
                batch_preds = model(batch_sequences).cpu().numpy()

                # Filter out any NaN predictions
                valid_mask = ~np.isnan(batch_preds).any(axis=1)
                predictions.extend(batch_preds[valid_mask])
                valid_targets.extend(val_targets[i:i + 32][valid_mask])
                valid_timestamps.extend([timestamps[j] for j, is_valid in
                                         enumerate(valid_mask, start=i) if is_valid])

        if not predictions:
            logger.error("No valid predictions generated")
            return None

        predictions = np.array(predictions).flatten()
        actual = np.array(valid_targets)

        # Calculate metrics
        self.metrics['mse'] = mean_squared_error(actual, predictions)
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['mae'] = mean_absolute_error(actual, predictions)
        self.metrics['r2'] = r2_score(actual, predictions)
        self.metrics['predictions'] = predictions
        self.metrics['actual_values'] = actual
        self.metrics['timestamps'] = valid_timestamps

        logger.info("Evaluation metrics:")
        logger.info(f"MSE: {self.metrics['mse']:.4f}")
        logger.info(f"RMSE: {self.metrics['rmse']:.4f}")
        logger.info(f"MAE: {self.metrics['mae']:.4f}")
        logger.info(f"R²: {self.metrics['r2']:.4f}")

        return self.metrics

    def generate_visualizations(self, save_dir: str = "visualizations"):
        """Generate and save all visualizations"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        self._plot_prediction_vs_actual(save_dir)
        self._plot_error_distribution(save_dir)
        self._plot_temporal_analysis(save_dir)
        self._plot_correlation_matrix(save_dir)
        self._plot_residuals(save_dir)

        # Save metrics to JSON
        with open(save_dir / 'metrics.json', 'w') as f:
            metrics_dict = {
                'mse': float(self.metrics['mse']),
                'rmse': float(self.metrics['rmse']),
                'mae': float(self.metrics['mae']),
                'r2': float(self.metrics['r2'])
            }
            json.dump(metrics_dict, f, indent=4)

    def _plot_prediction_vs_actual(self, save_dir: Path):
        """Plot predicted vs actual values"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics['actual_values'][:1000], label='Actual', alpha=0.7)
        plt.plot(self.metrics['predictions'][:1000], label='Predicted', alpha=0.7)
        plt.title('Predicted vs Actual Traffic Flow')
        plt.xlabel('Time Steps')
        plt.ylabel('Normalized Flow')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / 'prediction_vs_actual.png')
        plt.close()

    def _plot_error_distribution(self, save_dir: Path):
        """Plot error distribution"""
        errors = self.metrics['predictions'] - self.metrics['actual_values']
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title('Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(save_dir / 'error_distribution.png')
        plt.close()

    def _plot_temporal_analysis(self, save_dir: Path):
        """Plot temporal analysis of predictions"""
        timestamps = pd.to_datetime(self.metrics['timestamps'])

        # Daily patterns
        daily_df = pd.DataFrame({
            'hour': timestamps.hour,
            'error': np.abs(self.metrics['predictions'] - self.metrics['actual_values'])
        })
        daily_errors = daily_df.groupby('hour')['error'].mean()

        plt.figure(figsize=(12, 6))
        daily_errors.plot(kind='bar')
        plt.title('Average Prediction Error by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean Absolute Error')
        plt.tight_layout()
        plt.savefig(save_dir / 'temporal_analysis.png')
        plt.close()

    def _plot_correlation_matrix(self, save_dir: Path):
        """Plot correlation matrix of features"""
        # Load a sample of data for correlation analysis
        sample_file = next(self.data_dir.glob('**/*.csv'))
        df = pd.read_csv(sample_file)
        correlation = df[self.feature_cols].corr()

        plt.figure(figsize=(15, 12))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(save_dir / 'correlation_matrix.png')
        plt.close()

    def _plot_residuals(self, save_dir: Path):
        """Plot residuals analysis"""
        residuals = self.metrics['predictions'] - self.metrics['actual_values']

        plt.figure(figsize=(12, 6))
        plt.scatter(self.metrics['predictions'], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs Predicted Values')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.tight_layout()
        plt.savefig(save_dir / 'residuals.png')
        plt.close()

    def predict_future(self, last_sequence: np.ndarray, steps_ahead: int = 24) -> np.ndarray:
        """
        Predict traffic flow for specified number of steps ahead

        Args:
            last_sequence: Last known sequence of traffic data
            steps_ahead: Number of steps to predict into the future

        Returns:
            np.ndarray: Predicted values
        """
        model = TrafficLSTM(
            input_size=last_sequence.shape[1],
            hidden_size=128,
            num_layers=2,
            sequence_length=self.sequence_length
        ).to(self.device)

        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()

        predictions = []
        # Initialize current sequence with the last known sequence
        current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            for _ in range(steps_ahead):
                # Get prediction for next step
                next_pred = model(current_sequence)
                predictions.append(next_pred.cpu().numpy()[0])

                # Create a new sequence by shifting the window
                # Remove the oldest timestep and append the new prediction
                new_sequence = current_sequence.clone()
                # Move all steps one position back
                new_sequence[:, :-1, :] = current_sequence[:, 1:, :]
                # Add the new prediction in the correct format
                # We'll use the last known feature values and update only the prediction-relevant features
                new_sequence[:, -1, :] = current_sequence[:, -1, :]  # Copy last known features
                # Update the traffic flow feature (assuming it's in the correct position)
                traffic_flow_idx = self.feature_cols.index('Total_Flow_Normalized')
                new_sequence[:, -1, traffic_flow_idx] = next_pred.item()

                # Update current sequence for next iteration
                current_sequence = new_sequence

        return np.array(predictions)

    def plot_future_prediction(self, predictions: np.ndarray, save_dir: str = "predictions"):
        """Plot future predictions"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        plt.figure(figsize=(12, 6))
        plt.plot(predictions, marker='o')
        plt.title('Future Traffic Flow Predictions')
        plt.xlabel('Steps Ahead')
        plt.ylabel('Predicted Flow')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / 'future_predictions.png')
        plt.close()


class TrafficLSTM(torch.nn.Module):
    """LSTM model matching the training architecture"""

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.2, sequence_length=12):
        super(TrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        self.batch_norm_input = torch.nn.BatchNorm1d(sequence_length)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True, dropout=dropout)

        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.Linear(hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.batch_norm_input(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def main():
    """Main execution function with hardcoded paths and parameters"""
    # Define parameters here instead of using command-line arguments
    model_path = "F:/district_5_best_model.pth"  # Path to your .pth model file
    data_dir = "F:/TestSample"  # Path to your data directory
    save_dir = "analysis_results"  # Directory to save analysis results
    predict_ahead = 24  # Number of steps to predict into the future

    try:
        logger.info(f"Starting analysis with model: {model_path}")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Results will be saved to: {save_dir}")

        # Initialize analyzer with NaN handling
        analyzer = TrafficAnalyzer(model_path, data_dir)

        # Evaluate model
        logger.info("Evaluating model performance...")
        metrics = analyzer.evaluate_model()

        # Generate visualizations only if evaluation successful
        if metrics:
            logger.info("Generating visualizations...")
            analyzer.generate_visualizations(save_dir)

            # Make future predictions
            logger.info(f"Generating {predict_ahead}-step ahead predictions...")
            val_sequences, _, _ = analyzer.prepare_validation_data()

            if len(val_sequences) > 0:
                last_sequence = val_sequences[-1].numpy()
                future_predictions = analyzer.predict_future(last_sequence, predict_ahead)
                analyzer.plot_future_prediction(future_predictions, save_dir)

                # Save future predictions to CSV
                pd.DataFrame({
                    'step': range(1, predict_ahead + 1),
                    'predicted_flow': future_predictions.flatten()
                }).to_csv(Path(save_dir) / 'future_predictions.csv', index=False)

                # Generate comprehensive report
                logger.info("Generating analysis report...")
                generate_analysis_report(metrics, save_dir)

                logger.info(f"Analysis complete. Results saved to {save_dir}")
            else:
                logger.error("No valid sequences found in validation data")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


def generate_analysis_report(metrics: Dict, save_dir: str):
    """
    Generate a comprehensive analysis report in markdown format

    Args:
        metrics: Dictionary containing all computed metrics
        save_dir: Directory to save the report
    """
    save_dir = Path(save_dir)
    report_path = save_dir / 'analysis_report.md'

    with open(report_path, 'w') as f:
        f.write("# Traffic Prediction Model Analysis Report\n\n")

        # Model Performance Metrics
        f.write("## Performance Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean Squared Error (MSE) | {metrics['mse']:.4f} |\n")
        f.write(f"| Root Mean Squared Error (RMSE) | {metrics['rmse']:.4f} |\n")
        f.write(f"| Mean Absolute Error (MAE) | {metrics['mae']:.4f} |\n")
        f.write(f"| R² Score | {metrics['r2']:.4f} |\n\n")

        # Error Analysis
        f.write("## Error Analysis\n\n")
        errors = metrics['predictions'] - metrics['actual_values']
        f.write(f"- Mean Error: {np.mean(errors):.4f}\n")
        f.write(f"- Error Standard Deviation: {np.std(errors):.4f}\n")
        f.write(f"- Error Range: [{np.min(errors):.4f}, {np.max(errors):.4f}]\n\n")

        # Temporal Analysis
        f.write("## Temporal Analysis\n\n")
        timestamps = pd.to_datetime(metrics['timestamps'])
        hourly_errors = pd.DataFrame({
            'hour': timestamps.hour,
            'error': np.abs(errors)
        }).groupby('hour')['error'].mean()

        f.write("### Average Absolute Error by Hour\n\n")
        f.write("| Hour | Average Error |\n")
        f.write("|------|---------------|\n")
        for hour, error in hourly_errors.items():
            f.write(f"| {hour:02d}:00 | {error:.4f} |\n")
        f.write("\n")

        # Model Reliability
        f.write("## Model Reliability Analysis\n\n")

        # Calculate prediction intervals
        errors_95 = np.percentile(np.abs(errors), 95)
        errors_99 = np.percentile(np.abs(errors), 99)

        f.write("### Prediction Intervals\n\n")
        f.write(f"- 95% of predictions are within ±{errors_95:.4f} of actual values\n")
        f.write(f"- 99% of predictions are within ±{errors_99:.4f} of actual values\n\n")

        # Performance by Traffic Level
        f.write("## Performance by Traffic Level\n\n")
        traffic_levels = pd.qcut(metrics['actual_values'], q=5,
                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        level_metrics = pd.DataFrame({
            'actual': metrics['actual_values'],
            'predicted': metrics['predictions'],
            'traffic_level': traffic_levels
        })

        f.write("### Error Metrics by Traffic Level\n\n")
        f.write("| Traffic Level | RMSE | MAE | R² |\n")
        f.write("|--------------|------|-----|----|\n")

        for level in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
            level_data = level_metrics[level_metrics['traffic_level'] == level]
            level_rmse = np.sqrt(mean_squared_error(level_data['actual'], level_data['predicted']))
            level_mae = mean_absolute_error(level_data['actual'], level_data['predicted'])
            level_r2 = r2_score(level_data['actual'], level_data['predicted'])
            f.write(f"| {level} | {level_rmse:.4f} | {level_mae:.4f} | {level_r2:.4f} |\n")

        # Visualization References
        f.write("\n## Generated Visualizations\n\n")
        f.write("The following visualizations have been generated:\n\n")
        f.write("1. `prediction_vs_actual.png`: Comparison of predicted vs actual values\n")
        f.write("2. `error_distribution.png`: Distribution of prediction errors\n")
        f.write("3. `temporal_analysis.png`: Analysis of errors across time\n")
        f.write("4. `correlation_matrix.png`: Feature correlation analysis\n")
        f.write("5. `residuals.png`: Residuals analysis plot\n")
        f.write("6. `future_predictions.png`: Visualization of future predictions\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        # Based on R² score
        if metrics['r2'] < 0.5:
            f.write("- Consider model retraining with additional features or data\n")
            f.write("- Investigate potential data quality issues\n")
        elif metrics['r2'] < 0.7:
            f.write("- Model performance is moderate, consider feature engineering\n")
            f.write("- Evaluate the impact of seasonal patterns\n")
        else:
            f.write("- Model shows good predictive power\n")
            f.write("- Focus on maintaining data quality and regular retraining\n")

        # Based on error distribution
        if np.std(errors) > np.mean(np.abs(errors)):
            f.write("- High error variance suggests inconsistent predictions\n")
            f.write("- Consider ensemble methods or robust regression techniques\n")

        f.write("\n---\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
