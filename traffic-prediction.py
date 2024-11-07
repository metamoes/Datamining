import glob
import gc
import glob
import json
import logging
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Iterator, Optional
from typing import Union, Tuple, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

# Initialize basic logging until setup_logging is called
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def setup_logging(model_dir: Path, district: str):
    """
    Configure logging with both file and console handlers.

    Args:
        model_dir: Directory where logs will be stored
        district: Traffic district identifier

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # Create and configure file handler
    log_dir = model_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_dir / f'training_district_{district}_{timestamp}.log'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class ChunkedTrafficDataset(IterableDataset):
    """
    Memory-efficient dataset that streams data in chunks directly from disk.
    """

    def __init__(self, file_paths: List[str], sequence_length: int = 12,
                 chunk_size: int = 10000, shuffle: bool = True,
                 cache_dir: Optional[Path] = None):
        super().__init__()
        self.file_paths = file_paths
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Define feature groups
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
        self.target_col = 'Total_Flow_Normalized'

        # Calculate dataset properties
        self.input_size = len(self.feature_cols)  # Number of features
        self.total_samples = self._estimate_total_samples()

    def get_input_size(self) -> int:
        """Return the number of input features"""
        return self.input_size

    def _estimate_total_samples(self) -> int:
        """Estimate total number of samples without loading entire dataset"""
        total = 0
        for file_path in tqdm(self.file_paths, desc="Estimating dataset size"):
            try:
                file_size = Path(file_path).stat().st_size
                first_chunk = pd.read_csv(file_path, nrows=1000)
                avg_row_size = file_size / len(first_chunk)
                estimated_rows = int(file_size / avg_row_size)
                total += max(0, estimated_rows - self.sequence_length)
            except Exception as e:
                print(f"Warning: Could not estimate size for {file_path}: {e}")
        return total

    def _process_chunk(self, chunk: pd.DataFrame) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Process a single chunk of data into sequences"""
        try:
            # Basic preprocessing
            chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'])
            chunk = chunk.sort_values('Timestamp')

            # Extract features and handle missing values
            features_df = chunk[self.feature_cols].ffill().fillna(0)
            targets_series = chunk[self.target_col].ffill().fillna(0)

            data = features_df.values
            targets = targets_series.values

            # Create sequences
            for i in range(0, len(data) - self.sequence_length):
                x = data[i:i + self.sequence_length]
                y = targets[i + self.sequence_length - 1]

                if len(x) == self.sequence_length:
                    yield torch.FloatTensor(x), torch.FloatTensor([y])

        except Exception as e:
            print(f"Warning: Error processing chunk: {e}")
            return iter([])

    def _get_file_iterator(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Create an iterator over all files"""
        file_paths = self.file_paths.copy()
        if self.shuffle:
            np.random.shuffle(file_paths)

        for file_path in file_paths:
            try:
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                    yield from self._process_chunk(chunk)

                    # Clean up memory
                    del chunk
                    gc.collect()

            except Exception as e:
                print(f"Warning: Could not process file {file_path}: {e}")
                continue

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Return an iterator over the dataset"""
        return self._get_file_iterator()

    def __len__(self) -> int:
        """Return approximate length of dataset"""
        return self.total_samples


class StreamingDataLoader:
    """
    Custom data loader that handles streaming data with prefetching
    """

    def __init__(self, dataset: ChunkedTrafficDataset, batch_size: int = 32,
                 num_workers: int = 4, prefetch_factor: int = 2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def _create_batch(self, samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a batch from a list of samples"""
        sequences, targets = zip(*samples)
        return (
            torch.stack(sequences),
            torch.stack(targets)
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over batches"""
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self._create_batch(batch)
                batch = []

        # Yield last batch if not empty
        if batch:
            yield self._create_batch(batch)

    def __len__(self) -> int:
        """Return approximate number of batches"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class TrainingMetrics:
    """
    Tracks and manages training metrics, implementing early stopping and history tracking.

    Attributes:
        patience: Number of epochs to wait for improvement before stopping
        min_epochs: Minimum number of epochs to train
        max_epochs: Maximum number of epochs to train
        improvement_threshold: Minimum improvement required to consider as progress
        best_loss: Best validation loss achieved
        epochs_without_improvement: Counter for epochs without significant improvement
        training_history: Dictionary containing training metrics history
    """

    def __init__(self, patience: int = 10, min_epochs: int = 50,
                 max_epochs: int = 500, improvement_threshold: float = 0.001):
        self.patience = patience
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.improvement_threshold = improvement_threshold
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history: Dict[str, list] = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def update(self, epoch: int, train_loss: float, val_loss: float,
               learning_rate: float) -> bool:
        """
        Update metrics and determine if training should continue.

        Args:
            epoch: Current epoch number
            train_loss: Training loss for the epoch
            val_loss: Validation loss for the epoch
            learning_rate: Current learning rate

        Returns:
            bool: True if training should continue, False if it should stop
        """
        # Record metrics
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(float(train_loss))
        self.training_history['val_loss'].append(float(val_loss))
        self.training_history['learning_rate'].append(float(learning_rate))

        # Check if this is the best model so far
        if val_loss < (self.best_loss - self.improvement_threshold):
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            return True

        self.epochs_without_improvement += 1

        # Determine if training should continue
        if epoch < self.min_epochs:
            return True
        if epoch >= self.max_epochs:
            return False
        if self.epochs_without_improvement >= self.patience:
            return False

        return True

    def save_history(self, save_path: Path):
        """
        Save training history to JSON file.

        Args:
            save_path: Path where the history will be saved
        """
        with open(save_path, 'w') as f:
            json.dump(self.training_history, f, indent=4)


class TrafficDataset(Dataset):
    """
    Memory-efficient dataset that pre-processes traffic data in manageable chunks.

    Attributes:
        sequence_length: Length of input sequences
        target_col: Target column for prediction
        feature_cols: List of feature columns
        sequences: Processed input sequences
        targets: Processed target values
    """

    def __init__(self, file_paths, sequence_length=12, preprocessing_chunk_size=100000,
                 target_col='Total_Flow_Normalized'):
        self.sequence_length = sequence_length
        self.target_col = target_col

        # Define feature groups
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

        # Combine all features
        self.feature_cols = (
                self.temporal_features +
                self.road_features +
                self.traffic_features +
                self.lane_features
        )

        logger.info(f"Using {len(self.feature_cols)} features for prediction:")
        logger.info(f"Temporal features: {self.temporal_features}")
        logger.info(f"Road features: {self.road_features}")
        logger.info(f"Traffic features: {self.traffic_features}")
        logger.info(f"Lane features: {self.lane_features}")

        # Pre-process data in chunks
        sequences_list = []
        targets_list = []

        for file_path in tqdm(file_paths, desc="Loading data"):
            file_sequences, file_targets = self._process_file(file_path, preprocessing_chunk_size)
            if file_sequences is not None and file_targets is not None:
                sequences_list.extend(file_sequences)
                targets_list.extend(file_targets)

        if not sequences_list:
            raise ValueError("No valid sequences were created. Check data and sequence length.")

        logger.info(f"Concatenating {len(sequences_list)} sequences...")
        self.sequences = np.stack(sequences_list, axis=0)
        self.targets = np.array(targets_list).reshape(-1, 1)  # Reshape targets to 2D array
        logger.info(f"Final dataset shape - Sequences: {self.sequences.shape}, Targets: {self.targets.shape}")

    def _process_file(self, file_path: str, chunk_size: int):
        """Process a single file in chunks"""
        try:
            sequences_list = []
            targets_list = []
            chunks = pd.read_csv(file_path, chunksize=chunk_size)

            for chunk_idx, chunk in enumerate(chunks):
                # Verify all required columns are present
                missing_cols = set(self.feature_cols) - set(chunk.columns)
                if missing_cols:
                    logger.error(f"Missing columns in {file_path}: {missing_cols}")
                    continue

                # Basic preprocessing
                chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'])
                chunk = chunk.sort_values('Timestamp')

                # Extract features and handle missing values
                try:
                    features_df = chunk[self.feature_cols].ffill().fillna(0)
                    targets_series = chunk[self.target_col].ffill().fillna(0)

                    data = features_df.values
                    targets = targets_series.values
                except KeyError as e:
                    logger.error(f"Column access error in {file_path}: {str(e)}")
                    continue

                # Create sequences
                for i in range(0, len(data) - self.sequence_length, self.sequence_length):
                    x = data[i:i + self.sequence_length]
                    y = targets[i + self.sequence_length - 1]

                    if len(x) == self.sequence_length:
                        if x.shape[1] != len(self.feature_cols):
                            logger.warning(
                                f"Inconsistent features in {file_path}, "
                                f"chunk {chunk_idx}: expected {len(self.feature_cols)}, "
                                f"got {x.shape[1]}"
                            )
                            continue
                        sequences_list.append(x)
                        targets_list.append(y)

                # Clear memory
                del chunk
                gc.collect()

            return sequences_list, targets_list

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None, None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert to tensor and ensure correct shape
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor(self.targets[idx])
        return sequence, target


class TrafficLSTM(nn.Module):
    """
    LSTM model for traffic prediction with batch normalization and residual connections.

    Attributes:
        hidden_size: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        sequence_length: Length of input sequences
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, sequence_length=12):
        super(TrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # Input batch normalization
        self.batch_norm_input = nn.BatchNorm1d(sequence_length)

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        # Fully connected layers with batch normalization
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
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


class TrafficPredictor:
    """
    Main class for traffic prediction system.

    Attributes:
        data_dir: Directory containing traffic data
        model_dir: Directory for saving models and logs
        device: Device to use for training (CPU/GPU)
        batch_size: Size of training batches
        num_workers: Number of data loading workers
    """

    def __init__(self, data_dir, model_dir=None, device=None, batch_size=32, num_workers=4):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir) if model_dir else Path("models")
        self.device = device if device else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.models = {}
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create models directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Log initialization
        logger.info(f"Initialized TrafficPredictor with device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def _get_district_files(self, district):
        """Get all valid CSV files for a district with improved path handling"""
        patterns = [
            f"district_{district}/*.csv",
            f"district_{district}/**/*.csv",  # Recursive search
            f"**/district_{district}/*.csv",  # Search in subdirectories
            f"**/district_{district}/**/*.csv"  # Full recursive search
        ]

        all_files = []
        for pattern in patterns:
            files = glob.glob(str(self.data_dir / pattern), recursive=True)
            all_files.extend(files)

        all_files = sorted(list(set(all_files)))  # Remove duplicates and sort

        if not all_files:
            raise ValueError(f"No CSV files found for district {district} in {self.data_dir}")

        logger.info(f"Found {len(all_files)} files for district {district}")

        valid_files = []
        for file in all_files:
            try:
                pd.read_csv(file, nrows=1)  # Test file validity
                valid_files.append(file)
            except Exception as e:
                logger.warning(f"Skipping invalid file {file}: {str(e)}")

        if not valid_files:
            raise ValueError(f"No valid CSV files found for district {district}")

        logger.info(f"Successfully validated {len(valid_files)} files for district {district}")
        return valid_files

    def _split_data(self, files: List[str], val_ratio: float = 0.2) -> Tuple[List[str], List[str]]:
        """
        Split files into train and validation sets using modulo selection (every 5th file)
        and copy validation files to a separate directory.

        Args:
            files: List of file paths
            val_ratio: Ignored in this implementation as we use modulo selection

        Returns:
            Tuple[List[str], List[str]]: Lists of training and validation file paths
        """
        # Use modulo to select validation files (every 5th file)
        train_files = [f for i, f in enumerate(files) if i % 5 != 0]
        val_files = [f for i, f in enumerate(files) if i % 5 == 0]

        logger.info(f"Data split - Training files: {len(train_files)}, Validation files: {len(val_files)}")

        # Create validation directory for the district
        district = None
        for file in val_files:
            # Extract district number from file path
            match = re.search(r'district_(\d+)', file)
            if match:
                district = match.group(1)
                break

        if district:
            val_dir = Path(f"F:/PemsData/district_{district}/validation")
            val_dir.mkdir(parents=True, exist_ok=True)

            # Copy validation files to new directory
            logger.info(f"Copying validation files to {val_dir}")
            for file in tqdm(val_files, desc="Copying validation files"):
                # Get the original filename without the path
                filename = Path(file).name
                # Create new path in validation directory
                new_path = val_dir / filename
                try:
                    shutil.copy2(file, new_path)
                    logger.info(f"Copied {filename} to validation directory")
                except Exception as e:
                    logger.error(f"Failed to copy {filename}: {str(e)}")

        return train_files, val_files

    def validate_model(self, model, val_loader, criterion) -> float:
        """
        Run validation and return average loss

        Args:
            model: The neural network model
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            float: Average validation loss
        """
        model.eval()
        total_loss = 0
        batch_count = 0

        try:
            with torch.no_grad():
                for data, target in tqdm(val_loader, desc="Validation"):
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    total_loss += loss.item()
                    batch_count += 1

                    # Free up memory
                    del data, target, output
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise

        return total_loss / batch_count if batch_count > 0 else float('inf')

    def save_model(self, model, district, epoch, metrics, is_best=False):
        """
        Save model checkpoint with metadata

        Args:
            model: The neural network model
            district: District identifier
            epoch: Current epoch number
            metrics: Training metrics object
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = self.model_dir / f"district_{district}"
        checkpoint_dir.mkdir(exist_ok=True)

        if is_best:
            model_filename = f"district_{district}_best_model.pth"
        else:
            model_filename = f"district_{district}_epoch_{epoch}.pth"

        model_path = checkpoint_dir / model_filename

        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'training_metrics': {
                    'train_loss': metrics.training_history['train_loss'][-1],
                    'val_loss': metrics.training_history['val_loss'][-1],
                    'best_loss': metrics.best_loss,
                },
                'device': str(self.device),
                'district': district,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            torch.save(checkpoint, model_path)
            logger.info(f"Saved model checkpoint: {model_path}")

            # Save training history
            history_path = checkpoint_dir / f"training_history_{epoch}.json"
            metrics.save_history(history_path)
            logger.info(f"Saved training history: {history_path}")

        except Exception as e:
            logger.error(f"Error saving model checkpoint: {str(e)}")
            raise

    def _log_memory_usage(self):
        """Log current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")

    def train_district_model(self, district, min_epochs, max_epochs):
        """
        Train model for a specific district with fixed progress tracking and updated CUDA handling.
        """
        global logger
        logger = setup_logging(self.model_dir, district)

        try:
            # Get and split data files
            files = self._get_district_files(district)
            train_files, val_files = self._split_data(files)

            # Create datasets with optimized chunk sizes for GPU memory
            logger.info("Creating training dataset...")
            train_dataset = ChunkedTrafficDataset(
                train_files,
                sequence_length=12,
                chunk_size=5000,
                shuffle=True
            )
            total_train_samples = train_dataset.total_samples
            train_iterations = total_train_samples // 32  # based on batch_size
            logger.info(
                f"Training dataset created with {total_train_samples:,} samples ({train_iterations:,} iterations)")

            logger.info("Creating validation dataset...")
            val_dataset = ChunkedTrafficDataset(
                val_files,
                sequence_length=12,
                chunk_size=5000,
                shuffle=False
            )
            total_val_samples = val_dataset.total_samples
            val_iterations = total_val_samples // 64  # based on validation batch_size
            logger.info(
                f"Validation dataset created with {total_val_samples:,} samples ({val_iterations:,} iterations)")

            input_size = train_dataset.input_size
            logger.info(f"Input size: {input_size} features")

            # Initialize model and move to GPU
            model = TrafficLSTM(input_size=input_size).to(self.device)
            logger.info(f"Model initialized on {self.device}")

            # Create data loaders with prefetching
            logger.info("Creating data loaders...")
            train_loader = StreamingDataLoader(
                train_dataset,
                batch_size=32,
                num_workers=4,
                prefetch_factor=2
            )

            val_loader = StreamingDataLoader(
                val_dataset,
                batch_size=64,
                num_workers=2,
                prefetch_factor=2
            )

            # Optimizer with gradient clipping
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
            criterion = nn.MSELoss()

            # Updated scheduler initialization without verbose parameter
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            )

            metrics = TrainingMetrics(
                patience=10,
                min_epochs=min_epochs,
                max_epochs=max_epochs,
                improvement_threshold=0.0005
            )

            # Updated GradScaler initialization
            scaler = torch.amp.GradScaler('cuda')

            if torch.cuda.is_available():
                logger.info(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB allocated")

            # Estimate total training time
            estimated_seconds_per_epoch = (
                    (train_iterations * 0.1) +  # Estimated time per training iteration
                    (val_iterations * 0.05)  # Estimated time per validation iteration
            )
            estimated_total_time = time.strftime('%H:%M:%S', time.gmtime(estimated_seconds_per_epoch * max_epochs))
            logger.info(f"Estimated maximum training time: {estimated_total_time} (HH:MM:SS)")

            epoch = 0
            epoch_times = []

            while True:
                epoch += 1
                epoch_start_time = time.time()
                model.train()
                total_train_loss = 0
                batch_count = 0

                # Fixed progress bar initialization
                train_progress = tqdm(
                    total=train_iterations,
                    desc=f'Epoch {epoch}/{max_epochs}',
                    unit='batch',
                    dynamic_ncols=True,  # Automatically adjust to terminal width
                    leave=True
                )

                for batch_idx, (data, target) in enumerate(train_loader):
                    try:
                        data = data.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)

                        with torch.amp.autocast('cuda'):
                            output = model(data)
                            loss = criterion(output, target)

                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                        total_train_loss += loss.item()
                        batch_count += 1

                        # Update progress
                        current_loss = total_train_loss / batch_count
                        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0

                        train_progress.set_description(
                            f'Epoch {epoch}/{max_epochs} - Loss: {current_loss:.6f} - GPU: {gpu_mem:.1f}MB'
                        )
                        train_progress.update()

                        if batch_idx % 50 == 0:
                            del data, target, output, loss
                            torch.cuda.empty_cache()

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning(f"GPU OOM in batch {batch_idx}. Cleaning memory...")
                            del data, target, output, loss
                            torch.cuda.empty_cache()
                            continue
                        raise e

                train_progress.close()
                avg_train_loss = total_train_loss / batch_count if batch_count > 0 else float('inf')

                # Validation phase
                model.eval()
                total_val_loss = 0
                val_batch_count = 0

                val_progress = tqdm(
                    total=val_iterations,
                    desc='Validation',
                    unit='batch',
                    dynamic_ncols=True,
                    leave=True
                )

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader):
                        data = data.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)

                        with torch.amp.autocast('cuda'):
                            output = model(data)
                            loss = criterion(output, target)

                        total_val_loss += loss.item()
                        val_batch_count += 1

                        # Update validation progress
                        current_val_loss = total_val_loss / val_batch_count
                        val_progress.set_description(
                            f'Validation - Loss: {current_val_loss:.6f}'
                        )
                        val_progress.update()

                val_progress.close()
                avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else float('inf')

                # Calculate epoch time and estimates
                epoch_time = time.time() - epoch_start_time
                epoch_times.append(epoch_time)
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs = max_epochs - epoch
                estimated_remaining_time = time.strftime('%H:%M:%S', time.gmtime(avg_epoch_time * remaining_epochs))

                # Log epoch results
                logger.info(
                    f"Epoch {epoch}/{max_epochs} - "
                    f"Time: {time.strftime('%H:%M:%S', time.gmtime(epoch_time))} - "
                    f"Est. Remaining: {estimated_remaining_time} - "
                    f"Train Loss: {avg_train_loss:.6f} - "
                    f"Val Loss: {avg_val_loss:.6f} - "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} - "
                    f"GPU: {torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB"
                )

                scheduler.step(avg_val_loss)

                should_continue = metrics.update(
                    epoch, avg_train_loss, avg_val_loss,
                    optimizer.param_groups[0]['lr']
                )

                if avg_val_loss <= metrics.best_loss:
                    self.save_model(model, district, epoch, metrics, is_best=True)

                if epoch % 5 == 0:
                    self.save_model(model, district, epoch, metrics)

                if not should_continue:
                    total_training_time = time.strftime('%H:%M:%S', time.gmtime(sum(epoch_times)))
                    logger.info(
                        f"Training completed after {epoch} epochs in {total_training_time} - "
                        f"Best validation loss: {metrics.best_loss:.6f}"
                    )
                    break

                gc.collect()
                torch.cuda.empty_cache()

            return model

        except Exception as e:
            logger.error(f"Error training district {district}: {str(e)}")
            raise

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
                # Clean up resources
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

def get_user_input():
    """
    Get training parameters from user input with defaults.

    Returns:
        dict: Configuration parameters
    """
    print("\nTraffic Prediction System Configuration")
    print("-" * 40)

    # Default values
    defaults = {
        'district': '5',
        'min_epochs': 50,
        'max_epochs': 500,
        'data_dir': "E:\\Datamining-main\\pems_data\\processed_data"
    }

    config = {}

    # Get district
    district_input = input(f"\nEnter district number to train [default: {defaults['district']}]: ").strip()
    config['district'] = district_input if district_input else defaults['district']

    # Get minimum epochs
    while True:
        min_epochs_input = input(f"Enter minimum number of epochs [default: {defaults['min_epochs']}]: ").strip()
        if not min_epochs_input:
            config['min_epochs'] = defaults['min_epochs']
            break
        try:
            min_epochs = int(min_epochs_input)
            if min_epochs > 0:
                config['min_epochs'] = min_epochs
                break
            else:
                print("Minimum epochs must be greater than 0")
        except ValueError:
            print("Please enter a valid number")

    # Get maximum epochs
    while True:
        max_epochs_input = input(f"Enter maximum number of epochs [default: {defaults['max_epochs']}]: ").strip()
        if not max_epochs_input:
            config['max_epochs'] = defaults['max_epochs']
            break
        try:
            max_epochs = int(max_epochs_input)
            if max_epochs >= config['min_epochs']:
                config['max_epochs'] = max_epochs
                break
            else:
                print(f"Maximum epochs must be greater than or equal to minimum epochs ({config['min_epochs']})")
        except ValueError:
            print("Please enter a valid number")

    # Get data directory
    data_dir_input = input(f"Enter data directory path [default: {defaults['data_dir']}]: ").strip()
    config['data_dir'] = data_dir_input if data_dir_input else defaults['data_dir']

    # Display configuration
    print("\nConfiguration Summary:")
    print("-" * 20)
    print(f"District: {config['district']}")
    print(f"Minimum Epochs: {config['min_epochs']}")
    print(f"Maximum Epochs: {config['max_epochs']}")
    print(f"Data Directory: {config['data_dir']}")

    # Confirm configuration
    while True:
        confirm = input("\nProceed with this configuration? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            return config
        elif confirm in ['n', 'no']:
            print("\nRestarting configuration...")
            return get_user_input()
        else:
            print("Please enter 'y' or 'n'")


def main():
    """Main execution function with interactive configuration"""
    config = get_user_input()

    data_dir = Path(config['data_dir'])
    model_dir = Path("models")

    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    logger.info(f"Starting training with data from: {data_dir}")
    logger.info(f"Training configuration:")
    logger.info(f"  Minimum epochs: {config['min_epochs']}")
    logger.info(f"  Maximum epochs: {config['max_epochs']}")
    logger.info(f"  District: {config['district']}")

    predictor = TrafficPredictor(data_dir, model_dir=model_dir)
    predictor.train_districts(config['district'], config['min_epochs'], config['max_epochs'])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
