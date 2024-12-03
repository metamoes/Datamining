import argparse
import calendar
import gc
import json
import sys
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import colorama
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import redis
import torch
from colorama import Fore, Style
from logger import logger
from torch import nn
from tqdm import tqdm
from scipy import stats

def initialize_cli():
    """Initialize command line parser for basic setup"""
    parser = argparse.ArgumentParser(
        description='Traffic Prediction and Analysis System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to the trained model file'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing traffic data'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        default='cache',
        help='Directory for caching analysis results'
    )

    parser.add_argument(
        '--redis-host',
        type=str,
        default='localhost',
        help='Redis server host'
    )

    parser.add_argument(
        '--redis-port',
        type=int,
        default=6379,
        help='Redis server port'
    )

    return parser



def get_paths_interactively() -> tuple[str, str]:
    """Get model and data paths interactively"""
    try:
        logger.info("Starting interactive path collection")

        print(f"{Fore.CYAN}Please provide the path to your model file (.pth){Style.RESET_ALL}")
        while True:
            model_path = input(f"{Fore.GREEN}Model path: {Style.RESET_ALL}").strip()
            if not model_path:
                print(f"{Fore.RED}Path cannot be empty{Style.RESET_ALL}")
                continue

            path = Path(model_path)
            if not path.exists():
                print(f"{Fore.RED}File does not exist{Style.RESET_ALL}")
                continue
            if not path.is_file():
                print(f"{Fore.RED}Path is not a file{Style.RESET_ALL}")
                continue
            if path.suffix != '.pth':
                print(f"{Fore.RED}File must be a .pth file{Style.RESET_ALL}")
                continue
            break

        logger.info(f"Model path collected: {model_path}")

        print(f"\n{Fore.CYAN}Please provide the path to your data directory{Style.RESET_ALL}")
        while True:
            data_dir = input(f"{Fore.GREEN}Data directory path: {Style.RESET_ALL}").strip()
            if not data_dir:
                print(f"{Fore.RED}Path cannot be empty{Style.RESET_ALL}")
                continue

            path = Path(data_dir)
            if not path.exists():
                print(f"{Fore.RED}Directory does not exist{Style.RESET_ALL}")
                continue
            if not path.is_dir():
                print(f"{Fore.RED}Path is not a directory{Style.RESET_ALL}")
                continue
            break

        logger.info(f"Data directory collected: {data_dir}")

        return model_path, data_dir

    except Exception as e:
        logger.error(f"Error in get_paths_interactively: {str(e)}")
        raise


def get_setup_paths(default_model_path: str = "C:/NewProcessData/models/district_5/district_5_best_model.pth",
                    default_data_dir: str = "C:/NewProcessData/district_5",
                    default_save_dir: str = "analysis_results") -> tuple[str, str, str]:
    """Get paths from user or use defaults"""
    print("\nEnter paths (press Enter to use defaults):")

    model_path = input(f"Model path [{default_model_path}]: ").strip()
    model_path = model_path if model_path else default_model_path

    data_dir = input(f"Data directory [{default_data_dir}]: ").strip()
    data_dir = data_dir if data_dir else default_data_dir

    save_dir = input(f"Save directory [{default_save_dir}]: ").strip()
    save_dir = save_dir if save_dir else default_save_dir

    return model_path, data_dir, save_dir


def display_menu() -> int:
    """Display main menu and get user selection"""
    try:
        print(f"\n{Fore.CYAN}Traffic Analysis System{Style.RESET_ALL}")
        print("=" * 50)
        print("1. Predict traffic for specific date")
        print("2. Predict traffic for day of week")
        print("3. Analyze historical performance")
        print("4. Generate visualization suite")
        print("5. Compare with baseline")
        print("6. Generate full analysis report")
        print("0. Exit")
        print("=" * 50)

        while True:
            try:
                choice = input(f"{Fore.GREEN}Enter your choice (0-6): {Style.RESET_ALL}").strip()
                if not choice.isdigit():
                    print(f"{Fore.RED}Please enter a number.{Style.RESET_ALL}")
                    continue

                choice = int(choice)
                if 0 <= choice <= 6:
                    return choice
                print(f"{Fore.RED}Invalid choice. Please enter a number between 0 and 6.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Error in display_menu: {str(e)}")
        raise

def get_date_input() -> datetime:
    """Get date input from user"""
    try:
        while True:
            date_str = input(f"{Fore.GREEN}Enter date (YYYY-MM-DD) or 'today': {Style.RESET_ALL}").strip()
            if date_str.lower() == 'today':
                return datetime.now()
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                print(f"{Fore.RED}Invalid date format. Please use YYYY-MM-DD.{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Error in get_date_input: {str(e)}")
        raise


def get_day_of_week() -> str:
    """Get day of week input from user"""
    try:
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        print(f"\n{Fore.CYAN}Available days:{Style.RESET_ALL}")
        for idx, day in enumerate(days, 1):
            print(f"{idx}. {day}")

        while True:
            try:
                choice = input(f"{Fore.GREEN}Enter day number (1-7): {Style.RESET_ALL}").strip()
                if not choice.isdigit():
                    print(f"{Fore.RED}Please enter a number.{Style.RESET_ALL}")
                    continue

                choice = int(choice)
                if 1 <= choice <= 7:
                    return days[choice - 1]
                print(f"{Fore.RED}Invalid choice. Please enter a number between 1 and 7.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Error in get_day_of_week: {str(e)}")
        raise


class TrafficLSTM(nn.Module):
    """
    LSTM model for traffic prediction with batch normalization and residual connections.

    Attributes:
        hidden_size: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        sequence_length: Length of input sequences
    """

    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Larger model with additional layers
        self.batch_norm_input = nn.BatchNorm1d(12)  # sequence length

        # Bidirectional LSTM for better feature extraction
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Expanded fully connected layers
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),  # *2 for bidirectional
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        # Add value checking
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        # Input shape: (batch_size, sequence_length, features)
        batch_size = x.size(0)

        # Apply batch normalization to the sequence dimension
        x = self.batch_norm_input(x)

        # LSTM layers
        lstm_out, _ = self.lstm(x)

        # Take the last output and apply fully connected layers
        out = self.fc(lstm_out[:, -1, :])
        return out




    def train_districts(self, districts: Union[str, int, List[int]], min_epochs: int, max_epochs: int):
        """
        Train models for multiple districts.

        Args:
            districts: District identifier(s) to train
            min_epochs: Minimum number of epochs to train
            max_epochs: Maximum number of epochs to train
        """
        if districts == 'all':
            district_list = list(range(1, 11))
        elif isinstance(districts, (int, str)):
            district_list = [int(districts)]
        else:
            district_list = sorted(list(map(int, districts)))

        for district in district_list:
            try:
                logger.info(f"Starting training for district {district}")
                self.train_district_model(district, min_epochs, max_epochs)
                logger.info(f"Completed training for district {district}")
            except Exception as e:
                logger.error(f"Failed to train district {district}: {str(e)}")
                continue
            finally:
                torch.cuda.empty_cache()
                gc.collect()

class HolidayCalculator:
    """Calculates US Federal Holiday dates and manages holiday-specific analysis"""

    @staticmethod
    def get_nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
        """Get nth weekday of given month (e.g., 3rd Monday of January)"""
        c = calendar.monthcalendar(year, month)
        weekdays = [week[weekday] for week in c if week[weekday] != 0]
        return date(year, month, weekdays[n - 1])

    @staticmethod
    def get_last_monday_of_month(year: int, month: int) -> date:
        """Get last Monday of given month (for Memorial Day)"""
        c = calendar.monthcalendar(year, month)
        mondays = [week[calendar.MONDAY] for week in c if week[calendar.MONDAY] != 0]
        return date(year, month, mondays[-1])

    def get_federal_holidays(self, year: int) -> Dict[str, date]:
        """Calculate all federal holiday dates for given year"""
        holidays = {
            "New Year's Day": date(year, 1, 1),
            "Martin Luther King Jr. Day": self.get_nth_weekday_of_month(year, 1, calendar.MONDAY, 3),
            "Washington's Birthday": self.get_nth_weekday_of_month(year, 2, calendar.MONDAY, 3),
            "Memorial Day": self.get_last_monday_of_month(year, 5),
            "Juneteenth": date(year, 6, 19),
            "Independence Day": date(year, 7, 4),
            "Labor Day": self.get_nth_weekday_of_month(year, 9, calendar.MONDAY, 1),
            "Columbus Day": self.get_nth_weekday_of_month(year, 10, calendar.MONDAY, 2),
            "Veterans Day": date(year, 11, 11),
            "Thanksgiving Day": self.get_nth_weekday_of_month(year, 11, calendar.THURSDAY, 4),
            "Christmas Day": date(year, 12, 25)
        }
        return holidays

    def is_federal_holiday(self, check_date: date) -> Tuple[bool, Optional[str]]:
        """Check if given date is a federal holiday"""
        holidays = self.get_federal_holidays(check_date.year)
        for holiday_name, holiday_date in holidays.items():
            if check_date == holiday_date:
                return True, holiday_name
        return False, None


class TrafficAnalyzer:
    def __init__(self, model_path: str, data_dir: str, cache_dir: str = "cache",
                 redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize the Traffic Analyzer with model loading and feature configuration.

        Args:
            model_path: Path to the trained model file (.pth)
            data_dir: Directory containing traffic data files
            cache_dir: Directory for caching analysis results
            redis_host: Redis server hostname for caching
            redis_port: Redis server port number
        """
        # Set up basic configuration and paths
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.redis_host = redis_host
        self.redis_port = redis_port

        # Define temporal features that capture time-based patterns
        self.temporal_features = [
            'Hour',  # Hour of day (0-23)
            'Day_of_Week',  # Day of week (0-6, Monday=0)
            'Is_Weekend',  # Binary weekend indicator
            'Month',  # Month of year (1-12)
            'Is_Peak_Hour_Normalized'  # Binary peak hour indicator
        ]

        # Define road features that describe physical characteristics
        self.road_features = [
            'Station_Length_Normalized',  # Length of monitoring station
            'Active_Lanes_Normalized',  # Number of active lanes
            'Direction_S_Normalized',  # South direction indicator
            'Direction_E_Normalized'  # East direction indicator
        ]

        # Define overall traffic metrics
        self.traffic_features = [
            'Total_Flow_Normalized',  # Total vehicle flow rate
            'Avg_Occupancy_Normalized',  # Average road occupancy
            'Avg_Speed_Normalized'  # Average vehicle speed
        ]

        # Define per-lane metrics for all four lanes
        self.lane_features = []
        for lane in range(1, 5):  # Lanes 1-4
            self.lane_features.extend([
                f'Lane_{lane}_Flow_Normalized',  # Per-lane flow rate
                f'Lane_{lane}_Avg_Occ_Normalized',  # Per-lane occupancy
                f'Lane_{lane}_Avg_Speed_Normalized',  # Per-lane speed
                f'Lane_{lane}_Efficiency_Normalized'  # Per-lane efficiency
            ])

        # Combine all features into one list
        self.feature_cols = (
                self.temporal_features +
                self.road_features +
                self.traffic_features +
                self.lane_features
        )

        # Verify feature count matches model expectations
        expected_features = 28  # 5 temporal + 4 road + 3 traffic + (4 lanes × 4 metrics)
        total_features = len(self.feature_cols)

        if total_features != expected_features:
            feature_breakdown = (
                f"Feature count mismatch. Expected {expected_features} features, "
                f"but got {total_features}. Feature breakdown:\n"
                f"- Temporal features: {len(self.temporal_features)}\n"
                f"- Road features: {len(self.road_features)}\n"
                f"- Traffic features: {len(self.traffic_features)}\n"
                f"- Lane features: {len(self.lane_features)} "
                f"({len(self.lane_features) / 4} lanes × 4 metrics per lane)\n\n"
                f"Full feature list:\n"
            )
            for i, feature in enumerate(self.feature_cols, 1):
                feature_breakdown += f"{i}. {feature}\n"
            raise ValueError(feature_breakdown)

        # Initialize Redis connection with retry logic
        for attempt in range(3):
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=False,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()
                break
            except redis.ConnectionError as e:
                if attempt == 2:
                    raise ConnectionError(f"Failed to connect to Redis after 3 attempts: {e}")
                time.sleep(1)

        # Load the neural network model
        try:
            # Initialize model architecture
            self.model = TrafficLSTM(input_size=len(self.feature_cols)).to(self.device)

            # Load trained weights with safety flag
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True  # Safety measure for loading only weights
            )

            # Apply weights to model
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Set model to evaluation mode
            self.model.eval()

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Log successful initialization
        logger.info(f"TrafficAnalyzer initialized successfully on {self.device}")
        logger.info(f"Using {total_features} features for prediction")

    def predict_traffic(self, date_input, compare_baseline=True):
        """
        Generate traffic predictions with comprehensive statistical analysis.

        This method generates predictions and automatically calculates confidence intervals
        and other statistical measures needed for visualization and analysis.

        Args:
            date_input: Date for prediction (can be datetime or date object)
            compare_baseline: Whether to include historical baseline comparison

        Returns:
            Dictionary containing predictions and analysis results
        """
        try:
            target_date = date_input.date() if isinstance(date_input, datetime) else date_input
            logger.info(f"Preparing predictions for {target_date}")

            # Generate prediction intervals
            intervals = pd.date_range(
                start=datetime.combine(target_date, datetime.min.time()),
                end=datetime.combine(target_date, datetime.max.time()),
                freq='5min'
            )
            logger.info(f"Generated {len(intervals)} intervals")

            # Generate predictions
            predictions = []
            batch_size = 32

            with torch.no_grad():
                for i in range(0, len(intervals), batch_size):
                    batch_intervals = intervals[i:i + batch_size]
                    batch_predictions = []

                    for interval in batch_intervals:
                        sequence = self._prepare_sequence(interval)
                        output = self.model(sequence)
                        batch_predictions.append(output.cpu().item())

                    predictions.extend(batch_predictions)

            predictions = np.array(predictions)
            logger.info(f"Generated {len(predictions)} predictions")

            # Get baseline predictions if requested
            baseline_predictions = None
            if compare_baseline:
                baseline_predictions = self._get_baseline_predictions(target_date)

            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(predictions, baseline_predictions)

            # Generate analysis results
            analysis_results = self._analyze_predictions(predictions, baseline_predictions)

            return {
                'date': target_date,
                'intervals': intervals,
                'predictions': predictions,
                'baseline_predictions': baseline_predictions,
                'confidence_intervals': confidence_intervals,
                'analysis': analysis_results
            }

        except Exception as e:
            logger.error(f"Error in predict_traffic: {str(e)}")
            raise

    def _prepare_sequence(self, timestamp):
        """
        Prepare input sequence for prediction with proper CUDA handling.

        The key is to move the tensor to the GPU only once at the end of preparation,
        rather than moving individual pieces back and forth.
        """
        # Initialize sequence on CPU first - more efficient for numpy operations
        sequence = np.zeros((12, len(self.feature_cols)))

        # Generate temporal features for the last 12 5-minute intervals
        for i in range(12):
            # Calculate timestamp for this interval
            current_time = timestamp - pd.Timedelta(minutes=5 * (11 - i))

            # Set temporal features
            sequence[i, self.temporal_features.index('Hour')] = current_time.hour / 23.0
            sequence[i, self.temporal_features.index('Day_of_Week')] = current_time.weekday() / 6.0
            sequence[i, self.temporal_features.index('Is_Weekend')] = 1.0 if current_time.weekday() >= 5 else 0.0
            sequence[i, self.temporal_features.index('Month')] = current_time.month / 12.0
            sequence[i, self.temporal_features.index('Is_Peak_Hour_Normalized')] = (
                1.0 if (7 <= current_time.hour <= 9 or 16 <= current_time.hour <= 18) else 0.0
            )

            # Get historical patterns for non-temporal features
            historical_key = f"patterns:{current_time.strftime('%w:%H')}"
            historical_patterns = self.redis_client.get(historical_key)

            if historical_patterns:
                patterns = np.frombuffer(historical_patterns, dtype=np.float32)
                start_idx = len(self.temporal_features)
                sequence[i, start_idx:] = patterns
            else:
                remaining_features = self._get_default_patterns(current_time)
                start_idx = len(self.temporal_features)
                sequence[i, start_idx:] = remaining_features

        # Convert to tensor, add batch dimension, and move to GPU in one step
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        return sequence_tensor

    def _get_default_patterns(self, timestamp):
        """
        Generate default patterns for non-temporal features based on time of day.

        Args:
            timestamp: Current timestamp

        Returns:
            np.ndarray: Default feature values
        """
        # Initialize array for non-temporal features
        patterns = np.zeros(len(self.feature_cols) - len(self.temporal_features))
        current_pos = 0

        # Road features
        for _ in self.road_features:
            patterns[current_pos] = 0.5  # Default normalized value
            current_pos += 1

        # Traffic features
        hour = timestamp.hour
        is_peak = 7 <= hour <= 9 or 16 <= hour <= 18
        is_night = 22 <= hour or hour <= 5

        # Set traffic patterns based on time
        if is_peak:
            traffic_levels = [0.8, 0.7, 0.6]  # High traffic during peak
        elif is_night:
            traffic_levels = [0.2, 0.1, 0.9]  # Low traffic at night
        else:
            traffic_levels = [0.5, 0.4, 0.7]  # Moderate traffic otherwise

        for level in traffic_levels:
            patterns[current_pos] = level
            current_pos += 1

        # Lane features
        for _ in range(len(self.lane_features)):
            patterns[current_pos] = 0.5  # Default normalized value
            current_pos += 1

        return patterns

    def _get_baseline_predictions(self, target_date):
        """Get historical predictions for comparison"""
        # Try to get from cache first
        cache_key = f"baseline:{target_date.strftime('%Y%m%d')}"
        cached_baseline = self.redis_client.get(cache_key)

        if cached_baseline:
            return np.frombuffer(cached_baseline, dtype=np.float32)

        # Load historical data from files
        historical_files = list(self.data_dir.glob(f"**/*{target_date.strftime('%Y%m%d')}*.csv"))

        if historical_files:
            df = pd.read_csv(historical_files[0])
            baseline = df['Total_Flow_Normalized'].values

            # Cache for future use
            self.redis_client.setex(
                cache_key,
                timedelta(days=7),
                baseline.tobytes()
            )

            return baseline

        # If no historical data, use patterns based on day of week
        return self._generate_synthetic_baseline(target_date)

    def _generate_synthetic_baseline(self, target_date):
        """Generate synthetic baseline based on typical patterns"""
        intervals = pd.date_range(
            start=datetime.combine(target_date, datetime.min.time()),
            end=datetime.combine(target_date, datetime.max.time()),
            freq='5min'
        )

        baseline = []
        for interval in intervals:
            hour = interval.hour
            if 7 <= hour <= 9:  # Morning peak
                base_flow = 0.8 + np.random.normal(0, 0.05)
            elif 16 <= hour <= 18:  # Evening peak
                base_flow = 0.85 + np.random.normal(0, 0.05)
            elif 22 <= hour or hour <= 5:  # Night
                base_flow = 0.2 + np.random.normal(0, 0.02)
            else:  # Regular hours
                base_flow = 0.5 + np.random.normal(0, 0.03)

            baseline.append(max(0, min(1, base_flow)))  # Clip to [0, 1]

        return np.array(baseline)

    def _calculate_confidence_intervals(self, predictions: np.ndarray,
                                        baseline_predictions: Optional[np.ndarray]) -> Dict:
        """
        Calculate confidence intervals for predictions using rolling statistics.

        This provides a more robust confidence interval calculation by considering
        local variability in the predictions.

        Args:
            predictions: Array of traffic predictions
            baseline_predictions: Optional array of baseline predictions

        Returns:
            Dictionary containing confidence intervals at different levels
        """
        # Calculate rolling standard deviation (window of 12 = 1 hour)
        window_size = 12
        rolling_std = pd.Series(predictions).rolling(window=window_size, center=True).std()
        rolling_std = rolling_std.fillna(np.std(predictions))  # Fill edges with overall std

        intervals = {
            'model': {
                '95': {
                    'lower': predictions - 1.96 * rolling_std,
                    'upper': predictions + 1.96 * rolling_std
                },
                '99': {
                    'lower': predictions - 2.576 * rolling_std,
                    'upper': predictions + 2.576 * rolling_std
                }
            }
        }

        # Add baseline intervals if available
        if baseline_predictions is not None:
            baseline_rolling_std = pd.Series(baseline_predictions).rolling(
                window=window_size, center=True).std()
            baseline_rolling_std = baseline_rolling_std.fillna(np.std(baseline_predictions))

            intervals['baseline'] = {
                '95': {
                    'lower': baseline_predictions - 1.96 * baseline_rolling_std,
                    'upper': baseline_predictions + 1.96 * baseline_rolling_std
                }
            }

            # Add statistical comparison
            t_stat, p_value = stats.ttest_ind(predictions, baseline_predictions)
            intervals['comparison'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_difference': p_value < 0.05
            }

        return intervals

    def _analyze_predictions(self, predictions: np.ndarray, baseline_predictions: Optional[np.ndarray]) -> Dict:
        """
        Analyze predictions and generate comprehensive statistics.
        """
        analysis = {
            'model_stats': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            }
        }

        # Add peak analysis
        peaks = self._identify_peak_periods(predictions)
        analysis['model_stats']['peak_hours'] = peaks

        # Add baseline comparison if available
        if baseline_predictions is not None:
            analysis['baseline_stats'] = {
                'mean': float(np.mean(baseline_predictions)),
                'std': float(np.std(baseline_predictions)),
                'min': float(np.min(baseline_predictions)),
                'max': float(np.max(baseline_predictions))
            }

            # Calculate improvements
            mse = np.mean((predictions - baseline_predictions) ** 2)
            mae = np.mean(np.abs(predictions - baseline_predictions))

            analysis['improvements'] = {
                'mse': float(mse),
                'mae': float(mae),
                'percent_improvement': float(
                    (1 - mae / analysis['baseline_stats']['mean']) * 100
                )
            }

        return analysis

    def _identify_peak_periods(self, predictions: np.ndarray) -> Dict:
        """
        Identify morning and evening peak traffic periods with detailed analysis.

        This method analyzes traffic patterns to find peak periods by examining:
        1. Morning rush hour (typically 6 AM - 9 AM)
        2. Evening rush hour (typically 4 PM - 7 PM)
        3. Overall daily patterns for anomalous peaks

        Args:
            predictions: Array of traffic flow predictions for the day

        Returns:
            Dictionary containing peak period information including timing and flow rates
        """
        # Reshape predictions into hourly averages (288 5-minute intervals → 24 hours)
        hourly_values = predictions.reshape(-1, 12).mean(axis=1)

        # Define time windows for peak analysis
        morning_window = slice(6, 10)  # 6 AM to 9 AM
        evening_window = slice(16, 20)  # 4 PM to 7 PM

        # Analyze morning peak
        morning_traffic = hourly_values[morning_window]
        morning_peak_hour = morning_window.start + np.argmax(morning_traffic)
        morning_peak_flow = float(np.max(morning_traffic))

        # Analyze evening peak
        evening_traffic = hourly_values[evening_window]
        evening_peak_hour = evening_window.start + np.argmax(evening_traffic)
        evening_peak_flow = float(np.max(evening_traffic))

        # Calculate additional peak metrics
        peak_info = {
            'morning': {
                'time': f"{morning_peak_hour:02d}:00",
                'flow': morning_peak_flow,
                'duration': self._calculate_peak_duration(predictions, morning_peak_hour),
                'build_up_rate': self._calculate_build_up_rate(predictions, morning_peak_hour)
            },
            'evening': {
                'time': f"{evening_peak_hour:02d}:00",
                'flow': evening_peak_flow,
                'duration': self._calculate_peak_duration(predictions, evening_peak_hour),
                'build_up_rate': self._calculate_build_up_rate(predictions, evening_peak_hour)
            },
            'overall': {
                'dominant_peak': 'morning' if morning_peak_flow > evening_peak_flow else 'evening',
                'peak_ratio': float(morning_peak_flow / evening_peak_flow),
                'avg_peak_flow': float((morning_peak_flow + evening_peak_flow) / 2)
            }
        }

        return peak_info

    def _analyze_historical_performance(self, start_date: Optional[datetime] = None,
                                        end_date: Optional[datetime] = None) -> Dict:
        """
        Analyze historical traffic prediction performance over a specified time period.
        If no dates are provided, analyzes the last 30 days.

        Args:
            start_date: Beginning of analysis period (default: 30 days ago)
            end_date: End of analysis period (default: today)

        Returns:
            Dictionary containing comprehensive historical analysis
        """
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        logger.info(f"Analyzing historical performance from {start_date.date()} to {end_date.date()}")

        # Generate list of dates to analyze
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Collect predictions and metrics for each day
        daily_results = []
        daily_errors = []
        peak_accuracies = []

        for date in tqdm(date_range, desc="Analyzing historical data"):
            try:
                # Get predictions for this day
                results = self.predict_traffic(date, compare_baseline=True)
                predictions = results['predictions']
                baseline = results.get('baseline_predictions')

                # Calculate daily metrics
                daily_metrics = {
                    'date': date.date(),
                    'mean_flow': float(np.mean(predictions)),
                    'peak_flow': float(np.max(predictions)),
                    'min_flow': float(np.min(predictions))
                }

                # Calculate errors if baseline available
                if baseline is not None:
                    mae = np.mean(np.abs(predictions - baseline))
                    mse = np.mean((predictions - baseline) ** 2)
                    daily_errors.append({
                        'date': date.date(),
                        'mae': float(mae),
                        'mse': float(mse),
                        'rmse': float(np.sqrt(mse))
                    })

                # Analyze peak predictions
                peak_info = self._identify_peak_periods(predictions)
                peak_accuracies.append({
                    'date': date.date(),
                    'morning_peak': peak_info['morning']['flow'],
                    'evening_peak': peak_info['evening']['flow'],
                    'peak_ratio': peak_info['overall']['peak_ratio']
                })

                daily_results.append(daily_metrics)

            except Exception as e:
                logger.error(f"Error analyzing {date.date()}: {str(e)}")
                continue

        # Compile comprehensive analysis
        analysis = {
            'period': {
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'total_days': len(daily_results)
            },
            'daily_metrics': daily_results,
            'error_metrics': daily_errors,
            'peak_analysis': peak_accuracies,
            'summary_statistics': self._calculate_summary_statistics(daily_results, daily_errors)
        }

        return analysis

    def _calculate_summary_statistics(self, daily_metrics: List[Dict],
                                      error_metrics: List[Dict]) -> Dict:
        """Calculate summary statistics from historical analysis."""
        # Extract values for analysis
        mean_flows = [d['mean_flow'] for d in daily_metrics]
        peak_flows = [d['peak_flow'] for d in daily_metrics]

        summary = {
            'flow_statistics': {
                'average_daily_mean': float(np.mean(mean_flows)),
                'average_daily_peak': float(np.mean(peak_flows)),
                'flow_variability': float(np.std(mean_flows)),
                'peak_variability': float(np.std(peak_flows))
            }
        }

        # Add error statistics if available
        if error_metrics:
            maes = [e['mae'] for e in error_metrics]
            mses = [e['mse'] for e in error_metrics]
            summary['error_statistics'] = {
                'average_mae': float(np.mean(maes)),
                'average_mse': float(np.mean(mses)),
                'mae_variability': float(np.std(maes)),
                'best_day': min(error_metrics, key=lambda x: x['mae'])['date'],
                'worst_day': max(error_metrics, key=lambda x: x['mae'])['date']
            }

        return summary
    def _calculate_peak_duration(self, predictions: np.ndarray, peak_hour: int) -> float:
        """
        Calculate the duration of a peak period by finding how long traffic stays above 80% of peak.

        Args:
            predictions: Array of traffic predictions
            peak_hour: Hour when the peak occurs

        Returns:
            Duration of peak period in hours
        """
        # Convert peak hour to 5-minute interval index
        peak_interval = peak_hour * 12

        # Get the peak flow value
        peak_flow = predictions[peak_interval:peak_interval + 12].max()
        threshold = peak_flow * 0.8  # 80% of peak flow

        # Find intervals where flow is above threshold
        above_threshold = predictions >= threshold

        # Count consecutive intervals around peak
        start_idx = peak_interval
        while start_idx > 0 and above_threshold[start_idx - 1]:
            start_idx -= 1

        end_idx = peak_interval
        while end_idx < len(predictions) - 1 and above_threshold[end_idx + 1]:
            end_idx += 1

        # Convert intervals to hours
        duration = (end_idx - start_idx + 1) / 12  # 12 intervals per hour
        return float(duration)

    def _calculate_build_up_rate(self, predictions: np.ndarray, peak_hour: int) -> float:
        """
        Calculate how quickly traffic builds up to the peak.

        Args:
            predictions: Array of traffic predictions
            peak_hour: Hour when the peak occurs

        Returns:
            Rate of traffic increase (flow units per hour)
        """
        # Convert peak hour to 5-minute interval index
        peak_interval = peak_hour * 12

        # Look at 2 hours before peak
        start_interval = max(0, peak_interval - 24)

        # Calculate rate of change
        peak_flow = predictions[peak_interval:peak_interval + 12].max()
        start_flow = predictions[start_interval:start_interval + 12].mean()

        # Return hourly rate of change
        time_diff = (peak_interval - start_interval) / 12  # Convert to hours
        if time_diff > 0:
            rate = (peak_flow - start_flow) / time_diff
        else:
            rate = 0.0

        return float(rate)
    def _identify_peak_hours(self, values: np.ndarray) -> Dict:
        """Identify morning and evening peak hours"""
        # Reshape to 288 5-minute intervals
        intervals_per_day = 288
        if len(values) != intervals_per_day:
            values = values[:intervals_per_day]

        # Convert to hourly averages
        hourly_values = values.reshape(24, 12).mean(axis=1)

        # Find morning peak (5AM-11AM)
        morning_mask = slice(5, 11)
        morning_peak_hour = 5 + np.argmax(hourly_values[morning_mask])
        morning_peak_flow = float(np.max(hourly_values[morning_mask]))

        # Find evening peak (3PM-7PM)
        evening_mask = slice(15, 19)
        evening_peak_hour = 15 + np.argmax(hourly_values[evening_mask])
        evening_peak_flow = float(np.max(hourly_values[evening_mask]))

        return {
            'morning': {
                'time': f"{morning_peak_hour:02d}:00",
                'flow': morning_peak_flow
            },
            'evening': {
                'time': f"{evening_peak_hour:02d}:00",
                'flow': evening_peak_flow
            }
        }

    def _cache_predictions(self, target_date: date, results: Dict):
        """Cache prediction results in Redis with appropriate expiration"""
        cache_key = f"prediction:{target_date.strftime('%Y%m%d')}"

        cache_data = {
            'date': target_date.isoformat(),
            'predictions': results['predictions'].tobytes(),
            'intervals': [t.isoformat() for t in results['intervals']],
            'analysis': json.dumps(results['analysis'])
        }

        if results['baseline_predictions'] is not None:
            cache_data['baseline_predictions'] = results['baseline_predictions'].tobytes()

        # Store in Redis with 30-day expiration
        self.redis_client.hmset(cache_key, cache_data)
        self.redis_client.expire(cache_key, timedelta(days=30))

    def _decode_cached_prediction(self, cached_result: Dict) -> Dict:
        """Decode cached prediction results from Redis"""
        return {
            'date': datetime.fromisoformat(cached_result['date']).date(),
            'intervals': [datetime.fromisoformat(t) for t in json.loads(cached_result['intervals'])],
            'predictions': np.frombuffer(cached_result['predictions'], dtype=np.float32),
            'baseline_predictions': (
                np.frombuffer(cached_result['baseline_predictions'], dtype=np.float32)
                if 'baseline_predictions' in cached_result else None
            ),
            'analysis': json.loads(cached_result['analysis'])
        }

    def generate_prediction_report(self, prediction_results: Dict, save_dir: Path) -> Path:
        """
        Generate a comprehensive traffic prediction report with analysis and visualizations.

        Args:
            prediction_results: Dictionary containing prediction data
            save_dir: Directory where report will be saved

        Returns:
            Path: Path to the generated report file
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        report_path = save_dir / f"traffic_prediction_report_{prediction_results['date']}.md"

        # Ensure we have analysis data
        if 'analysis' not in prediction_results:
            prediction_results['analysis'] = self._analyze_predictions(
                prediction_results['predictions'],
                prediction_results.get('baseline_predictions')
            )

        # Generate visualizations if not already done
        vis_dir = save_dir / "visualizations"
        self._generate_visualization_suite(prediction_results, vis_dir)

        with open(report_path, 'w') as f:
            # Report Header
            f.write(f"# Traffic Prediction Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            self._write_executive_summary(f, prediction_results)

            # Add all report sections
            sections = [
                "## Detailed Analysis\n\n",
                "## Hourly Breakdown\n\n",
                "## Peak Period Analysis\n\n",
                "## Comparison with Historical Data\n\n",
                "## Visualizations\n\n",
                "## Recommendations\n\n",
                "## Technical Appendix\n\n"
            ]

            for section in sections:
                f.write(section)
                # Call appropriate writing method for each section
                method_name = f"_write_{section.strip('#').strip().lower().replace(' ', '_')}"
                if hasattr(self, method_name):
                    getattr(self, method_name)(f, prediction_results)
                else:
                    f.write("Section content to be generated.\n\n")

        return report_path

    def _write_executive_summary(self, f, results: Dict):
        """Write executive summary section of the report"""
        analysis = results['analysis']
        peaks = analysis['model_stats']['peak_hours']

        f.write("### Key Findings\n\n")
        f.write(f"- Average traffic flow: {analysis['model_stats']['mean']:.2f}\n")
        f.write(f"- Peak morning traffic: {peaks['morning']['flow']:.2f} at {peaks['morning']['time']}\n")
        f.write(f"- Peak evening traffic: {peaks['evening']['flow']:.2f} at {peaks['evening']['time']}\n")

        if 'baseline_stats' in analysis:
            improvement = analysis['improvements']['percent_improvement']
            f.write(f"- Performance vs historical: {improvement:+.1f}% difference\n")

    def _write_detailed_analysis(self, f, results: Dict):
        """Write detailed analysis section of the report"""
        analysis = results['analysis']

        f.write("### Traffic Flow Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean Flow | {analysis['model_stats']['mean']:.3f} |\n")
        f.write(f"| Standard Deviation | {analysis['model_stats']['std']:.3f} |\n")
        f.write(f"| Minimum Flow | {analysis['model_stats']['min']:.3f} |\n")
        f.write(f"| Maximum Flow | {analysis['model_stats']['max']:.3f} |\n")

        if 'confidence_intervals' in results:
            f.write("\n### Confidence Intervals\n\n")
            ci = results['confidence_intervals']['model']
            f.write("95% Confidence Interval:\n")
            f.write(f"- Lower bound: {np.mean(ci['95']['lower']):.3f}\n")
            f.write(f"- Upper bound: {np.mean(ci['95']['upper']):.3f}\n")

    def _write_hourly_breakdown(self, f, results: Dict):
        """Write hourly breakdown section of the report"""
        predictions = results['predictions']
        baseline = results['baseline_predictions']

        f.write("| Hour | Predicted Flow | Historical Flow | Difference |\n")
        f.write("|------|----------------|-----------------|------------|\n")

        for hour in range(24):
            # Get indices for this hour (12 5-minute intervals per hour)
            start_idx = hour * 12
            end_idx = start_idx + 12

            # Calculate average flows for this hour
            hour_pred = np.mean(predictions[start_idx:end_idx])
            hour_base = np.mean(baseline[start_idx:end_idx]) if baseline is not None else None

            if hour_base is not None:
                diff = hour_pred - hour_base
                f.write(f"| {hour:02d}:00 | {hour_pred:.3f} | {hour_base:.3f} | {diff:+.3f} |\n")
            else:
                f.write(f"| {hour:02d}:00 | {hour_pred:.3f} | N/A | N/A |\n")

    def _write_peak_analysis(self, f, results: Dict):
        """Write peak period analysis section of the report"""
        peaks = results['analysis']['model_stats']['peak_hours']

        f.write("### Morning Peak\n\n")
        f.write(f"- Time: {peaks['morning']['time']}\n")
        f.write(f"- Flow: {peaks['morning']['flow']:.3f}\n")

        f.write("\n### Evening Peak\n\n")
        f.write(f"- Time: {peaks['evening']['time']}\n")
        f.write(f"- Flow: {peaks['evening']['flow']:.3f}\n")

    def _write_baseline_comparison(self, f, results: Dict):
        """Write baseline comparison section of the report"""
        analysis = results['analysis']
        improvements = analysis['improvements']

        f.write("### Performance Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean Squared Error | {improvements['mse']:.6f} |\n")
        f.write(f"| Mean Absolute Error | {improvements['mae']:.6f} |\n")
        f.write(f"| Improvement | {improvements['percent_improvement']:+.2f}% |\n")

        if 'comparison' in results['confidence_intervals']:
            comp = results['confidence_intervals']['comparison']
            if comp['significant_difference']:
                f.write("\nThe difference from historical patterns is statistically significant ")
                f.write(f"(p = {comp['p_value']:.4f}).\n")

    def _write_visualization_section(self, f):
        """Write visualization section references"""
        f.write("The following visualizations have been generated:\n\n")
        f.write("1. Daily Traffic Flow Pattern (`traffic_flow.png`)\n")
        f.write("2. Peak Hours Analysis (`peak_hours.png`)\n")
        f.write("3. Comparison with Historical Data (`historical_comparison.png`)\n")
        f.write("4. Hourly Distribution (`hourly_distribution.png`)\n")

    def _write_recommendations(self, f, results: Dict):
        """Write recommendations based on analysis"""
        analysis = results['analysis']
        peaks = analysis['model_stats']['peak_hours']

        f.write("Based on the analysis, we recommend:\n\n")

        # Peak hour recommendations
        f.write(f"1. Plan for peak traffic at {peaks['morning']['time']} and {peaks['evening']['time']}\n")

        # Baseline comparison recommendations
        if 'improvements' in analysis:
            if analysis['improvements']['percent_improvement'] > 5:
                f.write("2. Maintain current traffic management strategies as they show improvement\n")
            elif analysis['improvements']['percent_improvement'] < -5:
                f.write("2. Review and adjust traffic management strategies to address decreased performance\n")

        # Variability recommendations
        if analysis['model_stats']['std'] > 0.2:
            f.write("3. Implement measures to reduce traffic flow variability\n")

    def _write_technical_appendix(self, f, results: Dict):
        """Write comprehensive technical appendix with detailed analysis parameters and methodology"""
        f.write("### Model Configuration\n\n")
        f.write("#### Core Parameters\n")
        f.write(f"- Model Type: Bidirectional LSTM with Attention\n")
        f.write(f"- Input Features: {len(self.feature_cols)}\n")
        f.write(f"- Sequence Length: 12 (1-hour historical window)\n")
        f.write(f"- Prediction Interval: 5 minutes\n")
        f.write(f"- Computing Device: {self.device}\n\n")

        f.write("#### Feature Groups\n\n")
        f.write("Temporal Features:\n")
        for feature in self.temporal_features:
            f.write(f"- {feature}\n")

        f.write("\nTraffic Features:\n")
        for feature in self.traffic_features:
            f.write(f"- {feature}\n")

        f.write("\nRoad Features:\n")
        for feature in self.road_features:
            f.write(f"- {feature}\n")

        f.write("\nLane-Specific Features:\n")
        for feature in self.lane_features:
            f.write(f"- {feature}\n")

        f.write("\n### Statistical Analysis Parameters\n\n")
        f.write("#### Confidence Intervals\n")
        f.write("- 95% Confidence Level: ±1.96 standard deviations\n")
        f.write("- 99% Confidence Level: ±2.576 standard deviations\n\n")

        f.write("#### Performance Metrics\n")
        if 'improvements' in results['analysis']:
            improvements = results['analysis']['improvements']
            f.write(f"- Mean Squared Error (MSE): {improvements['mse']:.6f}\n")
            f.write(f"- Mean Absolute Error (MAE): {improvements['mae']:.6f}\n")
            f.write(f"- Relative Improvement: {improvements['percent_improvement']:+.2f}%\n\n")

        f.write("### Data Processing Pipeline\n\n")
        f.write("1. Data Preprocessing\n")
        f.write("   - Outlier removal using IQR method\n")
        f.write("   - Missing value imputation using forward fill\n")
        f.write("   - Feature normalization to [0,1] range\n\n")

        f.write("2. Sequence Generation\n")
        f.write("   - Rolling window approach\n")
        f.write("   - 12-step historical sequences\n")
        f.write("   - 5-minute granularity\n\n")

        f.write("3. Model Architecture\n")
        f.write("   - Input Layer: Feature dimension matching\n")
        f.write("   - Batch Normalization\n")
        f.write("   - Bidirectional LSTM layers\n")
        f.write("   - Attention mechanism\n")
        f.write("   - Dense output layer\n\n")

        f.write("### Prediction Reliability\n\n")
        f.write("Confidence Score Calculation:\n")
        f.write("- Based on prediction variance\n")
        f.write("- Normalized to [0,1] range\n")
        f.write("- Weighted by historical accuracy\n\n")

        f.write("### Data Sources and Versions\n\n")
        f.write(f"- Model Checkpoint: {self.model_path.name}\n")
        f.write(f"- Data Directory: {self.data_dir.name}\n")
        f.write(f"- Cache Location: {self.cache_dir.name}\n")

    def _generate_visualization_suite(self, prediction_results: Dict, vis_dir: Path):
        """Generate complete visualization suite using only matplotlib."""
        # Use a professional matplotlib style
        plt.style.use('bmh')  # A clean, professional style available in matplotlib

        # Create visualizations directory
        vis_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating visualization suite...")

        # Calculate confidence intervals first if they don't exist
        if 'confidence_intervals' not in prediction_results:
            prediction_results['confidence_intervals'] = self._calculate_confidence_intervals(
                prediction_results['predictions'],
                prediction_results.get('baseline_predictions')
            )

        # Generate each visualization with proper error handling
        visualizations = [
            (self._plot_daily_pattern, "daily pattern plot"),
            (self._plot_peak_analysis, "peak analysis plot"),
            (self._plot_historical_comparison, "historical comparison plot"),
            (self._plot_hourly_heatmap, "hourly heatmap"),
            (self._plot_reliability_analysis, "reliability analysis plot")
        ]

        for plot_func, plot_name in visualizations:
            try:
                plot_func(prediction_results, vis_dir)
                logger.info(f"Generated {plot_name}")
            except Exception as e:
                logger.error(f"Error generating {plot_name}: {str(e)}")

    def _plot_daily_pattern(self, results: Dict, vis_dir: Path):
        """Create daily traffic pattern visualization."""
        plt.figure(figsize=(15, 8))

        # Convert timestamps to hours for x-axis
        hours = [(t.hour + t.minute / 60) for t in results['intervals']]

        # Plot predictions
        plt.plot(hours, results['predictions'],
                 color='#2C3E50',
                 label='Predicted Flow',
                 linewidth=2)

        # Plot confidence intervals
        ci = results['confidence_intervals']['model']['95']
        plt.fill_between(hours, ci['lower'], ci['upper'],
                         color='#2C3E50', alpha=0.2,
                         label='95% Confidence Interval')

        plt.title('Daily Traffic Flow Pattern', fontsize=14, pad=20)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Normalized Traffic Flow', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.savefig(vis_dir / 'traffic_flow.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_peak_analysis(self, results: Dict, vis_dir: Path):
        """Create peak period analysis visualization."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        hours = [(t.hour + t.minute / 60) for t in results['intervals']]
        predictions = results['predictions']

        # Morning peak (6 AM - 11 AM)
        morning_mask = [(6 <= t.hour <= 11) for t in results['intervals']]
        morning_hours = [h for h, m in zip(hours, morning_mask) if m]
        morning_flow = predictions[morning_mask]

        ax1.plot(morning_hours, morning_flow, color='#3498DB', linewidth=2)
        ax1.fill_between(morning_hours,
                         morning_flow * 0.9,
                         morning_flow * 1.1,
                         color='#3498DB', alpha=0.2)
        ax1.set_title('Morning Peak Analysis (6 AM - 11 AM)', fontsize=12)
        ax1.set_ylabel('Normalized Flow', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Evening peak (3 PM - 8 PM)
        evening_mask = [(15 <= t.hour <= 20) for t in results['intervals']]
        evening_hours = [h for h, m in zip(hours, evening_mask) if m]
        evening_flow = predictions[evening_mask]

        ax2.plot(evening_hours, evening_flow, color='#E67E22', linewidth=2)
        ax2.fill_between(evening_hours,
                         evening_flow * 0.9,
                         evening_flow * 1.1,
                         color='#E67E22', alpha=0.2)
        ax2.set_title('Evening Peak Analysis (3 PM - 8 PM)', fontsize=12)
        ax2.set_ylabel('Normalized Flow', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(vis_dir / 'peak_hours.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_historical_comparison(self, results: Dict, vis_dir: Path):
        """Create historical comparison visualization with deviation analysis"""
        if results['baseline_predictions'] is None:
            return

        plt.figure(figsize=(15, 10))

        # Set up two subplot panels
        gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        hours = [(t.hour + t.minute / 60) for t in results['intervals']]

        # Main comparison plot
        ax1.plot(hours, results['predictions'],
                 color='#2C3E50', label='Predicted', linewidth=2)
        ax1.plot(hours, results['baseline_predictions'],
                 color='#95A5A6', label='Historical', linewidth=2, alpha=0.7)

        # Shade the difference between predictions
        ax1.fill_between(hours,
                         results['predictions'],
                         results['baseline_predictions'],
                         where=results['predictions'] >= results['baseline_predictions'],
                         color='#27AE60', alpha=0.3, label='Improvement')
        ax1.fill_between(hours,
                         results['predictions'],
                         results['baseline_predictions'],
                         where=results['predictions'] < results['baseline_predictions'],
                         color='#E74C3C', alpha=0.3, label='Decline')

        ax1.set_title('Prediction vs Historical Comparison', fontsize=14, pad=20)
        ax1.set_ylabel('Normalized Flow', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Deviation plot
        deviation = results['predictions'] - results['baseline_predictions']
        ax2.plot(hours, deviation, color='#8E44AD', linewidth=2)
        ax2.fill_between(hours, deviation, 0,
                         where=deviation >= 0,
                         color='#27AE60', alpha=0.3)
        ax2.fill_between(hours, deviation, 0,
                         where=deviation < 0,
                         color='#E74C3C', alpha=0.3)

        ax2.set_title('Deviation Analysis', fontsize=12)
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Flow Difference', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.savefig(vis_dir / 'historical_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_hourly_heatmap(self, results: Dict, vis_dir: Path):
        """
        Create hourly traffic distribution heatmap using matplotlib.

        This visualization shows traffic patterns throughout the day using a color-coded
        heatmap, making it easy to identify peak periods and traffic intensity variations.
        """
        predictions = results['predictions']

        # Reshape data into hourly bins (24 hours x 12 5-minute intervals)
        hourly_data = predictions.reshape(24, 12)

        plt.figure(figsize=(15, 8))

        # Create heatmap using pcolormesh (more efficient than imshow for this case)
        plt.pcolormesh(hourly_data.T, cmap='YlOrRd')

        # Configure axes
        plt.xticks(np.arange(0.5, 24.5), [f'{i:02d}:00' for i in range(24)])
        plt.yticks(np.arange(0.5, 12.5), [f'{i * 5:02d}' for i in range(12)])

        # Add labels and title
        plt.title('Hourly Traffic Distribution', fontsize=14, pad=20)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Minutes Past Hour', fontsize=12)

        # Add colorbar
        plt.colorbar(label='Normalized Flow')

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save the figure
        plt.savefig(vis_dir / 'hourly_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_reliability_analysis(self, results: Dict, vis_dir: Path):
        """
        Create reliability analysis visualization using matplotlib's scatter plot.

        This visualization maps prediction reliability across the day, helping identify
        times when the model's predictions are most and least confident.
        """
        predictions = results['predictions']
        ci = results['confidence_intervals']['model']

        plt.figure(figsize=(15, 8))

        # Calculate reliability metrics
        hours = [(t.hour + t.minute / 60) for t in results['intervals']]
        uncertainty_width = ci['95']['upper'] - ci['95']['lower']
        reliability_score = 1 / (1 + uncertainty_width)

        # Create scatter plot
        scatter = plt.scatter(hours, predictions,
                              c=reliability_score,
                              cmap='RdYlGn',
                              s=50,
                              alpha=0.6)

        # Add colorbar
        plt.colorbar(scatter, label='Prediction Reliability Score')

        # Add trend line using numpy's polynomial fit
        z = np.polyfit(hours, predictions, 3)
        p = np.poly1d(z)
        plt.plot(hours, p(hours),
                 color='#404040',
                 linestyle='--',
                 alpha=0.8,
                 label='Trend')

        # Configure plot
        plt.title('Traffic Prediction Reliability Analysis', fontsize=14, pad=20)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Normalized Flow', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save the figure
        plt.savefig(vis_dir / 'reliability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_extended_visualization_suite(self, prediction_results: Dict, vis_dir: Path):
        """
        Generate extended suite of visualizations using only matplotlib.
        This creates additional analytical visualizations that provide deeper insights
        into traffic patterns and prediction reliability.
        """
        # Use a clean, professional matplotlib style
        plt.style.use('bmh')

        logger.info("Generating extended visualization suite...")

        # Define our visualization functions with descriptive names
        visualizations = [
            (self._plot_flow_distribution, "flow distribution analysis"),
            (self._plot_pattern_decomposition, "pattern decomposition"),
            (self._create_traffic_profile_clusters, "traffic profile clusters"),
            (self._plot_prediction_intervals_3d, "3D prediction landscape")
        ]

        # Generate each visualization with proper error handling
        for plot_func, plot_name in visualizations:
            try:
                plot_func(prediction_results, vis_dir)
                logger.info(f"Generated {plot_name}")
            except Exception as e:
                logger.error(f"Error generating {plot_name}: {str(e)}")

    def _generate_extended_visualization_suite(self, prediction_results: Dict, vis_dir: Path):
        """
        Generate extended suite of visualizations using only matplotlib.
        This creates additional analytical visualizations that provide deeper insights
        into traffic patterns and prediction reliability.
        """
        # Use a clean, professional matplotlib style
        plt.style.use('bmh')

        logger.info("Generating extended visualization suite...")

        # Define our visualization functions with descriptive names
        visualizations = [
            (self._plot_flow_distribution, "flow distribution analysis"),
            (self._plot_pattern_decomposition, "pattern decomposition"),
            (self._create_traffic_profile_clusters, "traffic profile clusters"),
            (self._plot_prediction_intervals_3d, "3D prediction landscape")
        ]

        # Generate each visualization with proper error handling
        for plot_func, plot_name in visualizations:
            try:
                plot_func(prediction_results, vis_dir)
                logger.info(f"Generated {plot_name}")
            except Exception as e:
                logger.error(f"Error generating {plot_name}: {str(e)}")

    def _plot_flow_distribution(self, results: Dict, vis_dir: Path):
        """
        Create flow distribution analysis using matplotlib's histogram functionality.
        This visualization shows how traffic flow varies during different periods
        of the day using overlapping histograms with transparency.
        """
        plt.figure(figsize=(15, 8))

        # Define time periods
        periods = {
            'Morning Peak': [(t.hour >= 6) and (t.hour <= 9) for t in results['intervals']],
            'Midday': [(t.hour > 9) and (t.hour < 16) for t in results['intervals']],
            'Evening Peak': [(t.hour >= 16) and (t.hour <= 19) for t in results['intervals']],
            'Night': [(t.hour > 19) or (t.hour < 6) for t in results['intervals']]
        }

        colors = ['#2ECC71', '#3498DB', '#E74C3C', '#9B59B6']

        # Create kernel density estimation manually for each period
        for (period, mask), color in zip(periods.items(), colors):
            period_data = results['predictions'][mask]

            # Create histogram with density normalization
            plt.hist(period_data, bins=30, density=True, alpha=0.3,
                     color=color, label=period)

            # Add smoothed line using numpy's histogram function
            counts, bins = np.histogram(period_data, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            # Smooth the line using a moving average
            smooth_counts = np.convolve(counts, np.ones(5) / 5, mode='same')
            plt.plot(bin_centers, smooth_counts, color=color, linewidth=2)

        plt.title('Traffic Flow Distribution by Time Period', fontsize=14, pad=20)
        plt.xlabel('Normalized Flow', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(vis_dir / 'flow_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pattern_decomposition(self, results: Dict, vis_dir: Path):
        """
        Create time series decomposition analysis using pure matplotlib.
        This visualization breaks down the traffic pattern into trend,
        seasonal, and residual components for deeper analysis.
        """
        plt.figure(figsize=(15, 12))

        # Convert predictions to time series
        values = results['predictions']
        n_samples = len(values)

        # Create subplots
        gs = plt.GridSpec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.4)

        # Original Data
        ax1 = plt.subplot(gs[0])
        ax1.plot(range(n_samples), values, color='#2C3E50')
        ax1.set_title('Original Time Series', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Calculate and plot trend using rolling mean
        window_size = 12  # One hour window
        trend = np.convolve(values, np.ones(window_size) / window_size, mode='same')
        ax2 = plt.subplot(gs[1])
        ax2.plot(range(n_samples), trend, color='#E74C3C')
        ax2.set_title('Trend Component', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Calculate and plot seasonal component
        seasonal = values - trend
        ax3 = plt.subplot(gs[2])
        ax3.plot(range(n_samples), seasonal, color='#3498DB')
        ax3.set_title('Seasonal Component', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # Calculate and plot residual component
        residual = values - trend - seasonal
        ax4 = plt.subplot(gs[3])
        ax4.plot(range(n_samples), residual, color='#2ECC71')
        ax4.set_title('Residual Component', fontsize=12)
        ax4.grid(True, alpha=0.3)

        plt.savefig(vis_dir / 'pattern_decomposition.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prediction_intervals_3d(self, results: Dict, vis_dir: Path):
        """
        Create a 3D visualization of predictions over time using a surface plot.
        This avoids triangulation issues by using a structured grid approach.
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create structured grid for 3D plot
        hours = np.array([(t.hour + t.minute / 60) for t in results['intervals']])
        predictions = results['predictions']

        # Create regular grid for surface plot
        X = np.linspace(0, 23, 24)  # Hours
        Y = np.linspace(min(predictions), max(predictions), 50)  # Flow values range
        X, Y = np.meshgrid(X, Y)

        # Calculate Z values (density of predictions at each point)
        Z = np.zeros_like(X)
        for i, x in enumerate(X[0]):
            mask = (hours >= x) & (hours < x + 1)
            if mask.any():
                hist, _ = np.histogram(predictions[mask], bins=50,
                                       range=(min(predictions), max(predictions)))
                Z[:, i] = hist

        # Normalize Z values
        Z = Z / Z.max()

        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                               linewidth=0, antialiased=True)

        # Customize the plot
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Traffic Flow')
        ax.set_zlabel('Relative Density')
        ax.set_title('3D Traffic Flow Distribution')

        # Add colorbar
        fig.colorbar(surf, ax=ax, label='Normalized Density')

        # Adjust view angle for better visualization
        ax.view_init(elev=30, azim=45)

        plt.savefig(vis_dir / 'prediction_3d.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_traffic_profile_clusters(self, results: Dict, vis_dir: Path):
        """
        Create clustered traffic profiles visualization using matplotlib.
        This visualization helps identify typical traffic patterns by
        grouping similar traffic behaviors together.
        """
        plt.figure(figsize=(15, 8))

        # Reshape data into hourly profiles (24 points per day)
        values = results['predictions']
        hourly_values = values.reshape(-1, 12).mean(axis=1)  # Convert to hourly averages

        # Create time axis
        hours = np.arange(24)

        # Plot average pattern
        plt.plot(hours, hourly_values,
                 color='#2C3E50',
                 linewidth=2,
                 label='Average Pattern')

        # Add confidence bands
        std_dev = np.std(values.reshape(-1, 12), axis=1)
        plt.fill_between(hours,
                         hourly_values - std_dev,
                         hourly_values + std_dev,
                         color='#2C3E50',
                         alpha=0.2,
                         label='±1 Standard Deviation')

        plt.title('Daily Traffic Profile', fontsize=14, pad=20)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Normalized Flow', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.savefig(vis_dir / 'traffic_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_historical_trends(self, analysis: Dict, vis_dir: Path):
        """
        Create visualization of historical traffic trends over time.
        Shows daily mean flow, peak flows, and trend lines for easy pattern recognition.

        Args:
            analysis: Dictionary containing historical analysis results
            vis_dir: Directory to save visualization
        """
        plt.figure(figsize=(15, 10))

        # Extract dates and metrics
        dates = [pd.to_datetime(metric['date']) for metric in analysis['daily_metrics']]
        mean_flows = [metric['mean_flow'] for metric in analysis['daily_metrics']]
        peak_flows = [metric['peak_flow'] for metric in analysis['daily_metrics']]

        # Plot mean and peak flows
        plt.plot(dates, mean_flows, color='#2C3E50', label='Mean Daily Flow', linewidth=2)
        plt.plot(dates, peak_flows, color='#E74C3C', label='Peak Daily Flow', linewidth=2)

        # Add trend lines
        z = np.polyfit(range(len(dates)), mean_flows, 1)
        p = np.poly1d(z)
        plt.plot(dates, p(range(len(dates))), '--', color='#2C3E50', alpha=0.5, label='Mean Flow Trend')

        z = np.polyfit(range(len(dates)), peak_flows, 1)
        p = np.poly1d(z)
        plt.plot(dates, p(range(len(dates))), '--', color='#E74C3C', alpha=0.5, label='Peak Flow Trend')

        plt.title('Historical Traffic Flow Trends', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Flow', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        plt.savefig(vis_dir / 'historical_trends.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_distribution(self, analysis: Dict, vis_dir: Path):
        """
        Create visualization of prediction error distribution over time.
        Updated to use current matplotlib parameter names.
        """
        plt.figure(figsize=(15, 10))

        # Extract error metrics if available
        if not analysis.get('error_metrics'):
            plt.text(0.5, 0.5, 'No error metrics available',
                     ha='center', va='center', fontsize=14)
            plt.savefig(vis_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        dates = [pd.to_datetime(metric['date']) for metric in analysis['error_metrics']]
        maes = [metric['mae'] for metric in analysis['error_metrics']]
        rmses = [metric['rmse'] for metric in analysis['error_metrics']]

        # Create box plots with updated parameter name
        plt.subplot(2, 1, 1)
        plt.boxplot([maes, rmses], tick_labels=['MAE', 'RMSE'])  # Updated parameter name
        plt.title('Error Distribution Statistics', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Plot error trends over time
        plt.subplot(2, 1, 2)
        plt.plot(dates, maes, color='#3498DB', label='MAE', linewidth=2)
        plt.plot(dates, rmses, color='#E67E22', label='RMSE', linewidth=2)
        plt.title('Error Metrics Over Time', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Error Magnitude', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(vis_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_peak_patterns(self, analysis: Dict, vis_dir: Path):
        """
        Create visualization of peak traffic patterns over time.
        Shows evolution of morning and evening peaks and their relationships.

        Args:
            analysis: Dictionary containing historical analysis results
            vis_dir: Directory to save visualization
        """
        plt.figure(figsize=(15, 10))

        # Extract peak data
        dates = [pd.to_datetime(metric['date']) for metric in analysis['peak_analysis']]
        morning_peaks = [metric['morning_peak'] for metric in analysis['peak_analysis']]
        evening_peaks = [metric['evening_peak'] for metric in analysis['peak_analysis']]
        peak_ratios = [metric['peak_ratio'] for metric in analysis['peak_analysis']]

        # Plot peak flows
        plt.subplot(2, 1, 1)
        plt.plot(dates, morning_peaks, color='#3498DB', label='Morning Peak', linewidth=2)
        plt.plot(dates, evening_peaks, color='#E67E22', label='Evening Peak', linewidth=2)
        plt.title('Peak Traffic Patterns Over Time', fontsize=14)
        plt.ylabel('Peak Flow', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)

        # Plot peak ratios
        plt.subplot(2, 1, 2)
        plt.plot(dates, peak_ratios, color='#2C3E50', linewidth=2)
        plt.axhline(y=1.0, color='#E74C3C', linestyle='--', alpha=0.5,
                    label='Equal Peaks Reference')
        plt.title('Morning/Evening Peak Ratio', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Ratio', fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(vis_dir / 'peak_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_historical_report(self, analysis: Dict, report_path: Path):
        """
        Generate a comprehensive report of historical traffic analysis.

        This method creates a detailed Markdown report that includes analysis of trends,
        patterns, and performance metrics over the specified time period. The report
        combines statistical analysis with insights about traffic patterns.

        Args:
            analysis: Dictionary containing historical analysis results
            report_path: Path where the report will be saved
        """
        with open(report_path, 'w') as f:
            # Report Header
            f.write("# Historical Traffic Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Analysis Period
            period = analysis['period']
            f.write("## Analysis Period\n\n")
            f.write(f"- Start Date: {period['start_date']}\n")
            f.write(f"- End Date: {period['end_date']}\n")
            f.write(f"- Total Days Analyzed: {period['total_days']}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = analysis['summary_statistics']
            f.write("### Traffic Flow Patterns\n\n")
            f.write(f"- Average Daily Flow: {summary['flow_statistics']['average_daily_mean']:.3f}\n")
            f.write(f"- Average Peak Flow: {summary['flow_statistics']['average_daily_peak']:.3f}\n")
            f.write(f"- Flow Variability: {summary['flow_statistics']['flow_variability']:.3f}\n\n")

            if 'error_statistics' in summary:
                f.write("### Prediction Performance\n\n")
                f.write(f"- Average MAE: {summary['error_statistics']['average_mae']:.3f}\n")
                f.write(f"- Average MSE: {summary['error_statistics']['average_mse']:.3f}\n")
                f.write(f"- Best Performance: {summary['error_statistics']['best_day']}\n")
                f.write(f"- Most Challenging Day: {summary['error_statistics']['worst_day']}\n\n")

            # Detailed Analysis
            f.write("## Detailed Analysis\n\n")

            # Traffic Flow Trends
            f.write("### Daily Traffic Flow Trends\n\n")
            f.write("| Date | Mean Flow | Peak Flow | Min Flow |\n")
            f.write("|------|-----------|-----------|----------|\n")
            for metric in analysis['daily_metrics']:
                f.write(f"| {metric['date']} | {metric['mean_flow']:.3f} | ")
                f.write(f"{metric['peak_flow']:.3f} | {metric['min_flow']:.3f} |\n")
            f.write("\n")

            # Peak Analysis
            f.write("### Peak Traffic Analysis\n\n")
            f.write("| Date | Morning Peak | Evening Peak | Peak Ratio |\n")
            f.write("|------|--------------|--------------|------------|\n")
            for peak in analysis['peak_analysis']:
                f.write(f"| {peak['date']} | {peak['morning_peak']:.3f} | ")
                f.write(f"{peak['evening_peak']:.3f} | {peak['peak_ratio']:.3f} |\n")
            f.write("\n")

            # Error Analysis if available
            if analysis.get('error_metrics'):
                f.write("### Prediction Error Analysis\n\n")
                f.write("| Date | MAE | MSE | RMSE |\n")
                f.write("|------|-----|-----|------|\n")
                for error in analysis['error_metrics']:
                    f.write(f"| {error['date']} | {error['mae']:.3f} | ")
                    f.write(f"{error['mse']:.3f} | {error['rmse']:.3f} |\n")
                f.write("\n")

            # Key Findings and Recommendations
            f.write("## Key Findings\n\n")
            findings = self._generate_key_findings(analysis)
            for finding in findings:
                f.write(f"- {finding}\n")
            f.write("\n")

            # Generated Visualizations
            f.write("## Generated Visualizations\n\n")
            f.write("The following visualizations have been generated:\n\n")
            f.write("1. `historical_trends.png`: Shows traffic flow trends over time\n")
            f.write("2. `error_distribution.png`: Displays prediction error patterns\n")
            f.write("3. `peak_patterns.png`: Illustrates peak traffic behavior\n\n")

            # Technical Details
            f.write("## Technical Details\n\n")
            f.write("### Analysis Parameters\n\n")
            f.write("- Data granularity: 5-minute intervals\n")
            f.write("- Metrics computed: Mean flow, peak flow, MAE, MSE, RMSE\n")
            f.write("- Peak detection window: 1 hour\n")

    def _generate_key_findings(self, analysis: Dict) -> List[str]:
        """
        Generate key findings from historical analysis.
        This method examines the data and extracts meaningful insights.
        """
        findings = []
        summary = analysis['summary_statistics']

        # Analyze overall trends
        flow_stats = summary['flow_statistics']
        avg_daily_mean = flow_stats['average_daily_mean']
        flow_variability = flow_stats['flow_variability']

        # Add findings about traffic patterns
        findings.append(
            f"Average daily traffic flow is {avg_daily_mean:.3f} with "
            f"variability of {flow_variability:.3f}"
        )

        # Analyze peak patterns
        peak_ratios = [p['peak_ratio'] for p in analysis['peak_analysis']]
        avg_ratio = np.mean(peak_ratios)
        if avg_ratio > 1.1:
            findings.append(
                f"Morning peaks tend to be stronger than evening peaks "
                f"by a factor of {avg_ratio:.2f}"
            )
        elif avg_ratio < 0.9:
            findings.append(
                f"Evening peaks tend to be stronger than morning peaks "
                f"by a factor of {1 / avg_ratio:.2f}"
            )

        # Add error analysis if available
        if 'error_statistics' in summary:
            error_stats = summary['error_statistics']
            findings.append(
                f"Prediction accuracy shows average MAE of "
                f"{error_stats['average_mae']:.3f}"
            )

        return findings


def day_to_num(day_name: str) -> int:
    """Convert day name to number (0-6 where Monday is 0)."""
    days = {
        'monday': 0,
        'tuesday': 1,
        'wednesday': 2,
        'thursday': 3,
        'friday': 4,
        'saturday': 5,
        'sunday': 6
    }
    return days[day_name.lower()]


def main():
    """
    Main CLI function with comprehensive traffic analysis capabilities.
    Provides interface for various traffic analysis operations including
    predictions, historical analysis, and report generation.
    """
    try:
        # Initialize logging and terminal colors
        logger.info("Starting Traffic Analysis CLI")
        colorama.init()

        # Get configuration paths with interactive input
        model_path, data_dir, save_dir = get_setup_paths()
        logger.info(f"Using paths - Model: {model_path}, Data: {data_dir}, Save: {save_dir}")

        # Initialize traffic analyzer with error handling
        logger.info("Initializing TrafficAnalyzer")
        try:
            analyzer = TrafficAnalyzer(
                model_path=model_path,
                data_dir=data_dir,
                cache_dir=save_dir,
                redis_host="localhost",
                redis_port=6379
            )
            print(f"{Fore.GREEN}Successfully initialized traffic analyzer!{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {str(e)}")
            print(f"{Fore.RED}Error initializing analyzer: {str(e)}{Style.RESET_ALL}")
            return 1

        # Main program loop for user interaction
        while True:
            try:
                choice = display_menu()

                if choice == 0:  # Exit
                    print(f"{Fore.YELLOW}Exiting...{Style.RESET_ALL}")
                    break

                elif choice == 1:  # Predict traffic for specific date
                    print(f"\n{Fore.CYAN}Traffic Prediction for Specific Date{Style.RESET_ALL}")
                    print("-" * 40)
                    date = get_date_input()
                    predictions = analyzer.predict_traffic(date, compare_baseline=False)
                    # Save simple prediction results
                    report_dir = Path(save_dir) / f"prediction_{date.strftime('%Y%m%d')}"
                    report_dir.mkdir(parents=True, exist_ok=True)
                    analyzer._plot_daily_pattern(predictions, report_dir)
                    print(f"\n{Fore.GREEN}Prediction saved to: {report_dir}{Style.RESET_ALL}")

                elif choice == 2:  # Predict traffic for day of week
                    print(f"\n{Fore.CYAN}Traffic Prediction for Day of Week{Style.RESET_ALL}")
                    print("-" * 40)
                    day = get_day_of_week()
                    # Calculate next occurrence of this day
                    today = datetime.now()
                    days_ahead = day_to_num(day) - today.weekday()
                    if days_ahead <= 0:
                        days_ahead += 7
                    target_date = today + timedelta(days=days_ahead)
                    predictions = analyzer.predict_traffic(target_date, compare_baseline=True)
                    report_dir = Path(save_dir) / f"prediction_{day.lower()}"
                    report_dir.mkdir(parents=True, exist_ok=True)
                    analyzer._plot_daily_pattern(predictions, report_dir)
                    print(f"\n{Fore.GREEN}Prediction saved to: {report_dir}{Style.RESET_ALL}")

                elif choice == 3:  # Analyze historical performance
                    print(f"\n{Fore.CYAN}Historical Performance Analysis{Style.RESET_ALL}")
                    print("-" * 40)

                    # Get date range with user interaction
                    print("\nEnter date range for analysis (press Enter to use last 30 days)")
                    start_date_str = input("Start date (YYYY-MM-DD): ").strip()
                    end_date_str = input("End date (YYYY-MM-DD or 'today'): ").strip()

                    try:
                        # Parse and validate dates
                        start_date = (datetime.strptime(start_date_str, '%Y-%m-%d')
                                      if start_date_str else None)
                        end_date = (datetime.now() if end_date_str.lower() == 'today' or not end_date_str
                                    else datetime.strptime(end_date_str, '%Y-%m-%d'))

                        print(f"\n{Fore.CYAN}Analyzing historical performance...{Style.RESET_ALL}")
                        analysis = analyzer._analyze_historical_performance(start_date, end_date)

                        # Create output directory structure
                        report_dir = Path(save_dir) / f"historical_analysis_{datetime.now().strftime('%Y%m%d')}"
                        report_dir.mkdir(parents=True, exist_ok=True)
                        vis_dir = report_dir / "visualizations"
                        vis_dir.mkdir(exist_ok=True)

                        # Generate visualizations and reports
                        print(f"\n{Fore.CYAN}Generating performance visualizations...{Style.RESET_ALL}")
                        analyzer._plot_historical_trends(analysis, vis_dir)
                        analyzer._plot_error_distribution(analysis, vis_dir)
                        analyzer._plot_peak_patterns(analysis, vis_dir)

                        report_path = report_dir / "historical_analysis_report.md"
                        analyzer._generate_historical_report(analysis, report_path)

                        print(f"\n{Fore.GREEN}Analysis complete!{Style.RESET_ALL}")
                        print(f"Report and visualizations saved to: {report_dir}")

                        # Offer to open results directory
                        if input("\nOpen report directory? (y/n): ").lower().startswith('y'):
                            import os
                            os.startfile(report_dir) if os.name == 'nt' else os.system(f'xdg-open {report_dir}')

                    except ValueError as e:
                        print(f"{Fore.RED}Error: Invalid date format. Please use YYYY-MM-DD.{Style.RESET_ALL}")
                        continue

                elif choice == 4:  # Generate visualization suite
                    print(f"\n{Fore.CYAN}Generating Visualization Suite{Style.RESET_ALL}")
                    print("-" * 40)
                    date = get_date_input()
                    predictions = analyzer.predict_traffic(date, compare_baseline=True)
                    vis_dir = Path(save_dir) / f"visualizations_{date.strftime('%Y%m%d')}"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    analyzer._generate_visualization_suite(predictions, vis_dir)
                    analyzer._generate_extended_visualization_suite(predictions, vis_dir)
                    print(f"\n{Fore.GREEN}Visualizations saved to: {vis_dir}{Style.RESET_ALL}")

                elif choice == 5:  # Compare with baseline
                    print(f"\n{Fore.CYAN}Baseline Comparison Analysis{Style.RESET_ALL}")
                    print("-" * 40)
                    date = get_date_input()
                    predictions = analyzer.predict_traffic(date, compare_baseline=True)
                    report_dir = Path(save_dir) / f"baseline_comparison_{date.strftime('%Y%m%d')}"
                    report_dir.mkdir(parents=True, exist_ok=True)
                    analyzer._plot_historical_comparison(predictions, report_dir)
                    print(f"\n{Fore.GREEN}Comparison saved to: {report_dir}{Style.RESET_ALL}")

                elif choice == 6:  # Generate full analysis report
                    print(f"\n{Fore.CYAN}Generating Full Analysis Report{Style.RESET_ALL}")
                    print("-" * 40)

                    # Get date and generate predictions
                    date = get_date_input()
                    print(f"\n{Fore.GREEN}Analyzing traffic for {date.strftime('%Y-%m-%d')}{Style.RESET_ALL}")

                    print(f"\n{Fore.CYAN}Generating predictions and comparisons...{Style.RESET_ALL}")
                    predictions = analyzer.predict_traffic(date, compare_baseline=True)

                    # Create report directory structure
                    report_dir = Path(save_dir) / f"report_{date.strftime('%Y%m%d')}"
                    report_dir.mkdir(parents=True, exist_ok=True)

                    # Generate visualizations
                    print(f"\n{Fore.CYAN}Creating visualization suite...{Style.RESET_ALL}")
                    vis_dir = report_dir / "visualizations"
                    vis_dir.mkdir(exist_ok=True)

                    analyzer._generate_visualization_suite(predictions, vis_dir)
                    analyzer._generate_extended_visualization_suite(predictions, vis_dir)

                    # Generate report
                    print(f"\n{Fore.CYAN}Generating comprehensive report...{Style.RESET_ALL}")
                    report_path = analyzer.generate_prediction_report(predictions, report_dir)

                    print(f"\n{Fore.GREEN}Analysis complete!{Style.RESET_ALL}")
                    print(f"Report and visualizations saved to: {report_dir}")
                    print("\nGenerated files:")
                    print(f"- Main report: {report_path.name}")
                    print("- Visualizations:")
                    for vis_file in vis_dir.glob("*.png"):
                        print(f"  - {vis_file.name}")

                    # Offer to open results directory
                    if input("\nOpen report directory? (y/n): ").lower().startswith('y'):
                        import os
                        os.startfile(report_dir) if os.name == 'nt' else os.system(f'xdg-open {report_dir}')

            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                print(f"{Fore.RED}An error occurred: {str(e)}{Style.RESET_ALL}")
                if not input("Continue? (y/n): ").lower().startswith('y'):
                    break

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"{Fore.RED}Fatal error: {str(e)}{Style.RESET_ALL}")
        return 1
    finally:
        colorama.deinit()


if __name__ == "__main__":
    sys.exit(main())
