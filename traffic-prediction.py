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
from typing import List, Iterator, Optional, Tuple, Union, Dict
import h5py
import numpy as np
import pandas as pd
import redis
import torch
import torch.nn as nn
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


class ValidationDebugger:
    """
    Utility class to debug validation issues including INF and NaN values.
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def check_tensor_values(self, tensor: torch.Tensor, name: str) -> Tuple[bool, Optional[str]]:
        """
        Check tensor for invalid values and basic statistics.
        Returns (is_valid, error_message)
        """
        if not torch.is_tensor(tensor):
            return False, f"{name} is not a tensor"

        try:
            # Check for NaN and INF
            if torch.isnan(tensor).any():
                return False, f"{name} contains NaN values"
            if torch.isinf(tensor).any():
                return False, f"{name} contains INF values"

            # Get statistics
            stats = {
                'min': float(tensor.min()),
                'max': float(tensor.max()),
                'mean': float(tensor.mean()),
                'std': float(tensor.std())
            }

            # Check for suspicious values
            if abs(stats['mean']) > 1e6 or stats['std'] > 1e6:
                return False, f"{name} has suspicious statistics: {stats}"

            self.logger.debug(f"{name} statistics: {stats}")
            return True, None

        except Exception as e:
            return False, f"Error checking {name}: {str(e)}"

    def debug_validation_batch(self, data: torch.Tensor, target: torch.Tensor,
                               output: torch.Tensor, loss: torch.Tensor) -> bool:
        """
        Debug a single validation batch.
        Returns True if batch is valid, False otherwise.
        """
        # Check input data
        valid, error = self.check_tensor_values(data, "Input data")
        if not valid:
            self.logger.error(error)
            return False

        # Check target values
        valid, error = self.check_tensor_values(target, "Target values")
        if not valid:
            self.logger.error(error)
            return False

        # Check model output
        valid, error = self.check_tensor_values(output, "Model output")
        if not valid:
            self.logger.error(error)
            return False

        # Check loss value
        if loss.item() == float('inf') or loss.item() == float('-inf'):
            self.logger.error(f"Loss is INF: {loss.item()}")
            return False
        if torch.isnan(loss):
            self.logger.error("Loss is NaN")
            return False

        return True

    def analyze_data_distribution(self, loader) -> dict:
        """
        Analyze the distribution of values in the dataset.
        """
        stats = {
            'data_min': float('inf'),
            'data_max': float('-inf'),
            'target_min': float('inf'),
            'target_max': float('-inf'),
            'total_samples': 0,
            'invalid_samples': 0
        }

        for data, target in loader:
            stats['total_samples'] += data.size(0)

            # Check for invalid values
            if torch.isnan(data).any() or torch.isinf(data).any():
                stats['invalid_samples'] += data.size(0)
                continue

            stats['data_min'] = min(stats['data_min'], float(data.min()))
            stats['data_max'] = max(stats['data_max'], float(data.max()))
            stats['target_min'] = min(stats['target_min'], float(target.min()))
            stats['target_max'] = max(stats['target_max'], float(target.max()))

        return stats


def validate_with_debugging(model: torch.nn.Module,
                            val_loader: torch.utils.data.DataLoader,
                            criterion: torch.nn.Module,
                            device: torch.device) -> float:
    """
    Enhanced validation function with debugging capabilities.
    """
    model.eval()
    debugger = ValidationDebugger()
    total_loss = 0
    batch_count = 0

    try:
        # Analyze dataset distribution first
        data_stats = debugger.analyze_data_distribution(val_loader)
        debugger.logger.info(f"Dataset statistics: {data_stats}")

        with torch.no_grad():
            for data, target in val_loader:
                try:
                    data, target = data.to(device), target.to(device)

                    # Run model with gradient hooks for debugging
                    output = model(data)
                    loss = criterion(output, target)

                    # Debug this batch
                    if not debugger.debug_validation_batch(data, target, output, loss):
                        debugger.logger.warning(f"Invalid batch detected at index {batch_count}")
                        continue

                    total_loss += loss.item()
                    batch_count += 1

                except RuntimeError as e:
                    debugger.logger.error(f"Runtime error in batch {batch_count}: {str(e)}")
                    continue
                finally:
                    # Clean up memory
                    del data, target, output
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if batch_count == 0:
            debugger.logger.error("No valid batches processed during validation")
            return float('inf')

        return total_loss / batch_count

    except Exception as e:
        debugger.logger.error(f"Validation failed: {str(e)}")
        return float('inf')


class ChunkedTrafficDataset(IterableDataset):
    """
    Memory-efficient dataset using Redis for metadata and HDF5 for sequence storage.
    Updated to handle the new data structure.
    """

    def __init__(self, file_paths: List[str], sequence_length: int = 12,
                 chunk_size: int = 10000, shuffle: bool = True,
                 cache_dir: Optional[Path] = None,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0):
        """Initialize with improved data validation"""
        super().__init__()
        self.file_paths = file_paths
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Sample first file to validate structure
        try:
            sample_df = pd.read_csv(file_paths[0], nrows=5)
            logger.info(f"Sample data shape: {sample_df.shape}")
            logger.info(f"Sample columns: {sample_df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error reading sample file: {str(e)}")
            raise

        # Updated feature groups based on the new data structure
        self.required_features = [
            'Is_Peak_Hour_Normalized',
            'Station_Length_Normalized',
            'Total_Flow_Normalized',
            'Avg_Occupancy_Normalized',
            'Avg_Speed_Normalized',
            'Direction_S_Normalized',
            'Direction_E_Normalized',
            'Direction_W_Normalized',
            'Lane_Type_FR_Normalized',
            'Lane_Type_ML_Normalized',
            'Lane_Type_OR_Normalized',
            'Active_Lanes_Normalized',
            'Lane_1_Flow_Normalized',
            'Lane_1_Avg_Occ_Normalized',
            'Lane_1_Avg_Speed_Normalized',
            'Lane_1_Efficiency_Normalized',
            'Lane_2_Flow_Normalized',
            'Lane_2_Avg_Occ_Normalized',
            'Lane_2_Avg_Speed_Normalized',
            'Lane_2_Efficiency_Normalized',
            'Lane_3_Flow_Normalized',
            'Lane_3_Avg_Occ_Normalized',
            'Lane_3_Avg_Speed_Normalized',
            'Lane_3_Efficiency_Normalized',
            'Lane_4_Flow_Normalized',
            'Lane_4_Avg_Occ_Normalized',
            'Lane_4_Avg_Speed_Normalized',
            'Lane_4_Efficiency_Normalized'
        ]

        # Additional temporal features
        self.temporal_features = [
            'Hour',
            'Day_of_Week',
            'Is_Weekend',
            'Month'
        ]

        # Combined feature list
        self.feature_cols = self.required_features

        # Validate features exist in data
        missing_features = set(self.feature_cols) - set(sample_df.columns)
        if missing_features:
            logger.error(f"Missing required features in data: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")

        self.target_col = 'Total_Flow_Normalized'
        if self.target_col not in sample_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")

        # Log feature configuration
        logger.info(f"Using {len(self.feature_cols)} features for prediction")
        logger.info(f"Features: {self.feature_cols}")

        # Set up Redis connection with retry logic
        for attempt in range(3):
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
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

        # HDF5 path for sequence storage
        self.h5_path = self.cache_dir / "sequences.h5"

        # Input size for the model (28 features)
        self.input_size = len(self.feature_cols)
        logger.info(f"Input size: {self.input_size} features")

        # Initialize storage and process data
        self._init_storage()
        self._process_files()

        # Get total samples
        try:
            total_samples = self.redis_client.get('total_samples')
            self.total_samples = int(total_samples) if total_samples else 0
            if self.total_samples <= 0:
                raise ValueError("No valid samples found in dataset")
            logger.info(f"Total samples in dataset: {self.total_samples:,}")
        except Exception as e:
            logger.error(f"Error getting total samples: {e}")
            raise

    def _validate_sequence(self, sequence: np.ndarray, target: float) -> bool:
        """Validate a single sequence"""
        try:
            # Basic shape check
            if sequence.shape[0] != self.sequence_length:
                return False

            # Check for required features
            if sequence.shape[1] != len(self.feature_cols):
                return False

            # Check for NaN/Inf values
            if np.isnan(sequence).any() or np.isnan(target):
                return False
            if np.isinf(sequence).any() or np.isinf(target):
                return False

            # Ensure all values are finite
            if not np.all(np.isfinite(sequence)) or not np.isfinite(target):
                return False

            return True

        except Exception as e:
            logger.error(f"Error in sequence validation: {str(e)}")
            return False

    def _process_chunk(self, chunk: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Process chunk with proper data reading and validation"""
        sequences = []
        targets = []

        try:
            # Log chunk info
            logger.info(f"Processing chunk with shape: {chunk.shape}")

            # Verify required columns
            missing_cols = set(self.feature_cols) - set(chunk.columns)
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.info(f"Available columns: {chunk.columns.tolist()}")
                return [], []

            # Extract features and targets
            features_df = chunk[self.feature_cols].copy()
            targets_series = chunk[self.target_col].copy()

            # Convert to numeric, replacing any non-numeric values with NaN
            features_df = features_df.apply(pd.to_numeric, errors='coerce')
            targets_series = pd.to_numeric(targets_series, errors='coerce')

            # Handle missing values
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            targets_series = targets_series.fillna(method='ffill').fillna(method='bfill').fillna(0)

            # Convert to numpy arrays
            data = features_df.values.astype(np.float32)
            targets_data = targets_series.values.astype(np.float32)

            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Target shape: {targets_data.shape}")

            # Create sequences
            for i in range(len(data) - self.sequence_length + 1):
                sequence = data[i:(i + self.sequence_length)]
                target = targets_data[i + self.sequence_length - 1]

                if self._validate_sequence(sequence, target):
                    sequences.append(sequence)
                    targets.append(np.array([target]))

            logger.info(f"Created {len(sequences)} valid sequences")

            if not sequences:
                logger.warning("No valid sequences created. Sample data:")
                logger.warning(features_df.head())
                logger.warning("Sample statistics:")
                logger.warning(features_df.describe())

        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return [], []

        return sequences, targets

    def _validate_sequence_with_reason(self, sequence: np.ndarray, target: float) -> Tuple[bool, str]:
        """Validate sequence with detailed reason for failure"""
        try:
            # Shape check
            if sequence.shape[0] != self.sequence_length:
                return False, f"Wrong sequence length: {sequence.shape[0]} vs expected {self.sequence_length}"

            # Feature count check
            if sequence.shape[1] != len(self.feature_cols):
                return False, f"Wrong feature count: {sequence.shape[1]} vs expected {len(self.feature_cols)}"

            # NaN check
            if np.isnan(sequence).any():
                nan_count = np.isnan(sequence).sum()
                return False, f"Contains {nan_count} NaN values in sequence"
            if np.isnan(target):
                return False, "NaN target value"

            # Inf check
            if np.isinf(sequence).any():
                inf_count = np.isinf(sequence).sum()
                return False, f"Contains {inf_count} infinite values in sequence"
            if np.isinf(target):
                return False, "Infinite target value"

            # Value range check
            sequence_min = np.min(sequence)
            sequence_max = np.max(sequence)
            if sequence_min < -10 or sequence_max > 10:
                return False, f"Values out of expected range: min={sequence_min}, max={sequence_max}"

            return True, "valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _validate_arrays(self, sequences: np.ndarray, targets: np.ndarray) -> bool:
        """Validate arrays before writing to HDF5"""
        try:
            # Check shapes
            if sequences.shape[0] != targets.shape[0]:
                logger.error("Sequence and target counts don't match")
                return False

            if sequences.shape[1] != self.sequence_length:
                logger.error(f"Invalid sequence length: {sequences.shape[1]}")
                return False

            if sequences.shape[2] != self.input_size:
                logger.error(f"Invalid feature count: {sequences.shape[2]}, expected {self.input_size}")
                return False

            # Check for NaN/inf values
            if np.isnan(sequences).any() or np.isnan(targets).any():
                logger.error("Arrays contain NaN values")
                return False

            if np.isinf(sequences).any() or np.isinf(targets).any():
                logger.error("Arrays contain infinite values")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating arrays: {str(e)}")
            return False

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Improved iterator with proper indexing and error handling.
        """
        worker_info = torch.utils.data.get_worker_info()

        try:
            with h5py.File(self.h5_path, 'r') as f:
                total_sequences = f['sequences'].shape[0]

                # Generate all indices first
                if worker_info is None:
                    # Single worker case
                    indices = np.arange(total_sequences)
                else:
                    # Multiple worker case - split indices among workers
                    per_worker = total_sequences // worker_info.num_workers
                    worker_id = worker_info.id
                    start_idx = worker_id * per_worker
                    end_idx = start_idx + per_worker if worker_id < worker_info.num_workers - 1 else total_sequences
                    indices = np.arange(start_idx, end_idx)

                # Shuffle if needed
                if self.shuffle:
                    np.random.shuffle(indices)

                # Process in chunks with sorted access
                chunk_size = 1000  # Adjust this based on memory constraints
                for chunk_start in range(0, len(indices), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(indices))
                    chunk_indices = indices[chunk_start:chunk_end]

                    # Sort indices for HDF5 access
                    sorted_indices = np.sort(chunk_indices)

                    # Load data
                    sequences = f['sequences'][sorted_indices]
                    targets = f['targets'][sorted_indices]

                    # Create index mapping to restore shuffled order
                    restore_order = np.argsort(np.argsort(chunk_indices))

                    # Yield sequences in original shuffled order
                    for idx in range(len(chunk_indices)):
                        original_idx = restore_order[idx]
                        seq = sequences[original_idx]
                        target = targets[original_idx]

                        if np.isfinite(seq).all() and np.isfinite(target).all():
                            # Ensure correct shapes
                            seq = seq.reshape(self.sequence_length, -1)
                            target = target.reshape(-1)

                            # Convert to tensors
                            seq_tensor = torch.FloatTensor(seq)
                            target_tensor = torch.FloatTensor(target)

                            yield seq_tensor, target_tensor

        except Exception as e:
            logger.error(f"Error in dataset iterator: {str(e)}")
            raise

    def __len__(self) -> int:
        """Return total number of samples"""
        return self.total_samples

    def _init_storage(self):
        """Initialize storage with Redis cleanup"""
        try:
            # Clear ALL Redis keys to force reprocessing
            self.redis_client.flushdb()
            logger.info("Cleared Redis database")

            # Initialize HDF5 file with minimal compression
            with h5py.File(self.h5_path, 'w') as f:
                f.create_dataset('sequences',
                                 shape=(0, self.sequence_length, self.input_size),
                                 maxshape=(None, self.sequence_length, self.input_size),
                                 chunks=(100, self.sequence_length, self.input_size),
                                 dtype='float32',
                                 compression='gzip',
                                 compression_opts=1)

                f.create_dataset('targets',
                                 shape=(0, 1),
                                 maxshape=(None, 1),
                                 chunks=(100, 1),
                                 dtype='float32',
                                 compression='gzip',
                                 compression_opts=1)

            logger.info("Storage initialized successfully")
            logger.info(f"Sequence length: {self.sequence_length}")
            logger.info(f"Feature count: {self.input_size}")

        except Exception as e:
            logger.error(f"Error initializing storage: {str(e)}")
            raise

    def _write_buffer_to_hdf5(self, sequence_buffer: List[np.ndarray],
                              target_buffer: List[np.ndarray],
                              current_size: int):
        """Write buffer to HDF5 with detailed debug logging"""
        if not sequence_buffer or not target_buffer:
            logger.warning("Empty buffer received, skipping HDF5 write")
            return

        try:
            logger.info(f"Writing buffer - Sequences: {len(sequence_buffer)}, Current size: {current_size}")

            with h5py.File(self.h5_path, 'a') as f:
                # Convert lists to arrays with validation
                sequences_array = np.stack(sequence_buffer)
                targets_array = np.stack(target_buffer)

                logger.info(
                    f"Arrays created - Sequences shape: {sequences_array.shape}, Targets shape: {targets_array.shape}")

                # Validate arrays
                if not self._validate_arrays(sequences_array, targets_array):
                    return

                # Calculate new size
                new_size = current_size + len(sequences_array)
                logger.info(f"Resizing datasets to: {new_size}")

                # Resize and write with detailed error handling
                try:
                    f['sequences'].resize(new_size, axis=0)
                    f['targets'].resize(new_size, axis=0)

                    f['sequences'][current_size:new_size] = sequences_array
                    f['targets'][current_size:new_size] = targets_array

                    logger.info(f"Successfully wrote {len(sequences_array)} sequences")

                except Exception as e:
                    logger.error(f"Error during HDF5 write operation: {str(e)}")
                    logger.error(
                        f"Current dataset shapes - Sequences: {f['sequences'].shape}, Targets: {f['targets'].shape}")
                    raise

        except Exception as e:
            logger.error(f"Error in HDF5 write operation: {str(e)}")
            raise

    def _process_files(self):
        """Process files with improved data reading"""
        pipe = self.redis_client.pipeline()
        total_sequences = 0
        sequence_buffer = []
        target_buffer = []
        buffer_size = 10000

        try:
            logger.info("\nStarting file processing...")
            logger.info(f"Total files to process: {len(self.file_paths)}")

            for file_idx, file_path in enumerate(tqdm(self.file_paths, desc="Processing files")):
                try:
                    # Read CSV with comma separator
                    df = pd.read_csv(file_path)
                    logger.info(f"\nProcessing {Path(file_path).name}")
                    logger.info(f"Initial shape: {df.shape}")
                    logger.info(f"Columns: {df.columns.tolist()}")

                    # Process in chunks
                    chunk_size = self.chunk_size
                    for i in range(0, len(df), chunk_size):
                        chunk = df.iloc[i:i + chunk_size].copy()
                        sequences, targets = self._process_chunk(chunk)

                        if sequences:
                            sequence_buffer.extend(sequences)
                            target_buffer.extend(targets)

                            # Write when buffer is full
                            if len(sequence_buffer) >= buffer_size:
                                self._write_buffer_to_hdf5(sequence_buffer, target_buffer, total_sequences)
                                total_sequences += len(sequence_buffer)
                                sequence_buffer = []
                                target_buffer = []

                    logger.info(f"Processed file {Path(file_path).name}: Total sequences so far: {total_sequences}")

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    continue

                # Execute Redis commands periodically
                if file_idx % 5 == 0:
                    pipe.execute()

            # Write remaining data
            if sequence_buffer:
                self._write_buffer_to_hdf5(sequence_buffer, target_buffer, total_sequences)
                total_sequences += len(sequence_buffer)

            # Save total count
            pipe.set('total_samples', total_sequences)
            pipe.execute()

            logger.info(f"\nProcessing completed:")
            logger.info(f"Total sequences created: {total_sequences}")

            if total_sequences == 0:
                raise ValueError("No valid sequences were created")

        except Exception as e:
            logger.error(f"Error in file processing: {str(e)}")
            raise

    def _verify_input_file(self, file_path: Path) -> bool:
        """Verify input file format and content"""
        try:
            # Read the first few rows
            df = pd.read_csv(file_path, sep='\t', nrows=5)

            # Log file info
            logger.info(f"\nVerifying file: {file_path.name}")
            logger.info(f"File size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")
            logger.info(f"Column count: {len(df.columns)}")
            logger.info(f"Available columns: {df.columns.tolist()}")

            # Check for required columns
            missing_cols = set(self.feature_cols) - set(df.columns)
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False

            # Check data types
            logger.info("\nColumn data types:")
            for col in self.feature_cols:
                dtype = df[col].dtype
                logger.info(f"{col}: {dtype}")
                if not np.issubdtype(dtype, np.number):
                    logger.error(f"Non-numeric data in column {col}")
                    return False

            # Check for NaN values
            nan_cols = df[self.feature_cols].isna().sum()
            if nan_cols.any():
                logger.warning(f"NaN values found in columns: {nan_cols[nan_cols > 0]}")

            return True

        except Exception as e:
            logger.error(f"Error verifying file {file_path}: {str(e)}")
            return False


class StreamingDataLoader:
    """
    Custom data loader with improved batch handling and memory management
    """
    def __init__(self, dataset: ChunkedTrafficDataset, batch_size: int = 32,
                 num_workers: int = 4, prefetch_factor: int = 2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = False  # Keep partial batches

    def _create_batch(self, samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a batch with improved error handling and validation"""
        if not samples:
            raise ValueError("Empty batch received")

        try:
            sequences, targets = zip(*samples)
            batch_sequences = torch.stack(sequences)
            batch_targets = torch.stack(targets)

            # Validate batch values with improved handling
            batch_sequences = torch.nan_to_num(batch_sequences, nan=0.0, posinf=1e6, neginf=-1e6)
            batch_targets = torch.nan_to_num(batch_targets, nan=0.0, posinf=1e6, neginf=-1e6)

            return batch_sequences, batch_targets

        except Exception as e:
            logger.error(f"Error creating batch: {e}")
            raise

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over batches with improved error handling"""
        batch = []
        for sample in self.dataset:
            try:
                batch.append(sample)
                if len(batch) == self.batch_size:
                    yield self._create_batch(batch)
                    batch = []
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                batch = []  # Reset batch on error
                continue

        # Handle last batch if not empty
        if batch and not self.drop_last:
            try:
                yield self._create_batch(batch)
            except Exception as e:
                logger.error(f"Error processing final batch: {e}")

    def __len__(self) -> int:
        """Return number of batches"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
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
        and copy validation files to a separate directory if not already present.

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

        # Extract district number from file path
        district = None
        for file in val_files:
            match = re.search(r'district_(\d+)', file)
            if match:
                district = match.group(1)
                break

        if district:
            val_dir = Path(f"E:/PemsData/district_{district}/validation")
            val_dir.mkdir(parents=True, exist_ok=True)

            # Check which files need to be copied
            files_to_copy = []
            for file in val_files:
                filename = Path(file).name
                target_path = val_dir / filename

                if not target_path.exists():
                    files_to_copy.append((file, target_path))
                else:
                    logger.info(f"Validation file already exists, skipping: {filename}")

            if files_to_copy:
                logger.info(f"Copying {len(files_to_copy)} new validation files to {val_dir}")
                for file, target_path in tqdm(files_to_copy, desc="Copying validation files"):
                    try:
                        shutil.copy2(file, target_path)
                        logger.info(f"Copied {Path(file).name} to validation directory")
                    except Exception as e:
                        logger.error(f"Failed to copy {Path(file).name}: {str(e)}")
            else:
                logger.info("All validation files already present in target directory")

            # Update val_files to point to the validation directory
            val_files = [str(val_dir / Path(f).name) for f in val_files]

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
                avg_val_loss = validate_with_debugging(
    model,
    val_loader,
    criterion,
    self.device
)

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
