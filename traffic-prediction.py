import configparser
import inspect
import logging
import os
import queue
import time
import traceback
from pathlib import Path
from threading import Thread
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
else:
    print(f"\nCUDA is available on device: {torch.cuda.get_device_name(0)}\n")
    torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)


def verify_paths(config):
    """Verify that all required paths exist and are accessible"""
    base_path = Path(config.config['PATHS']['data_dir'])

    # Check if base path exists
    if not base_path.exists():
        logging.error(f"Base directory does not exist: {base_path}")
        raise FileNotFoundError(f"Base directory not found: {base_path}")

    logging.info(f"Base directory found: {base_path}")

    # Print directory contents for debugging
    logging.info("Directory contents:")
    for item in base_path.glob('*'):
        logging.info(f"Found: {item}")

    return base_path

class Config:
    """Configuration manager for the traffic prediction system"""

    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser(comment_prefixes=';', allow_no_value=True)
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            self.create_default_config()
        self.config.read(self.config_path)

    def create_default_config(self):
        config = self.config

        config['DEFAULT'] = {
            '; Traffic Prediction System Configuration': None
        }

        config['PATHS'] = {
            '; Base directory for all data': None,
            'data_dir': 'E:/Datamining-main/pems_data/raw_data',
            '; Directory for saved models': None,
            'model_dir': 'models',
            '; Directory for checkpoints': None,
            'checkpoint_dir': 'checkpoints'
        }

        config['MODEL'] = {
            '; Number of input features': None,
            'input_size': '12',
            '; Hidden layer sizes (comma-separated)': None,
            'hidden_layers': '64,32',
            '; Output size (number of predictions)': None,
            'output_size': '3',
            '; Dropout rate for regularization': None,
            'dropout': '0.2'
        }

        config['TRAINING'] = {
            '; Number of epochs': None,
            'epochs': '100',
            '; Batch size for training': None,
            'batch_size': '32',
            '; Learning rate': None,
            'learning_rate': '0.001',
            '; Checkpoint save frequency (epochs)': None,
            'checkpoint_frequency': '5'
        }

        with open(self.config_path, 'w') as f:
            config.write(f)


class DataPreprocessor:
    """Handles efficient data preprocessing and caching with safe loading"""

    def __init__(self, district_path):
        self.district_path = Path(district_path)
        self.cache_path = self.district_path / 'preprocessed_cache.pt'
        logging.info(f"Initializing DataPreprocessor for {district_path}")

    def preprocess_and_cache(self):
        if self.cache_path.exists():
            cache_size = self.cache_path.stat().st_size / (1024 * 1024)  # Size in MB
            logging.info(f"Found existing cache file ({cache_size:.2f} MB)")
            logging.info(f"Cache file location: {self.cache_path}")
            start_time = time.time()
            try:
                # Use safe loading with weights_only=True
                data = torch.load(self.cache_path, weights_only=True, map_location='cpu')
                logging.info(f"Cache loaded in {time.time() - start_time:.2f} seconds")
                return data
            except Exception as e:
                logging.warning(f"Error loading cached data: {str(e)}")
                logging.info("Regenerating cache file...")
                return self._generate_cache()
        else:
            return self._generate_cache()

    def _generate_cache(self):
        """Generate new cache file from raw data"""
        logging.info("Starting data preprocessing...")
        data_files = list(self.district_path.glob('d03_text_station_5min_*.txt'))
        if not data_files:
            raise FileNotFoundError(f"No data files found in {self.district_path}")

        logging.info(f"Found {len(data_files)} data files to process")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in data_files)
        logging.info(f"Total data size: {total_size / (1024 * 1024):.2f} MB")

        # Count total rows first
        logging.info("Counting total rows...")
        total_rows = 0
        for f in tqdm(data_files, desc="Counting rows"):
            total_rows += len(pd.read_csv(f, header=None))
        logging.info(f"Total rows to process: {total_rows:,}")

        # Pre-allocate memory
        logging.info("Pre-allocating memory for features...")
        all_features = np.zeros((total_rows, 3), dtype=np.float32)

        # Process files with detailed progress
        current_idx = 0
        processed_rows = 0
        start_time = time.time()

        for file_path in tqdm(data_files, desc="Processing files"):
            try:
                file_start = time.time()
                df = pd.read_csv(file_path, header=None, usecols=[8, 9, 10])
                chunk_size = len(df)
                all_features[current_idx:current_idx + chunk_size] = df.values
                current_idx += chunk_size
                processed_rows += chunk_size

                file_time = time.time() - file_start
                rows_per_sec = chunk_size / file_time
                logging.info(
                    f"Processed {file_path.name}: {chunk_size:,} rows in {file_time:.2f}s ({rows_per_sec:.0f} rows/s)")

                if processed_rows % 100000 == 0:
                    progress = processed_rows / total_rows * 100
                    elapsed = time.time() - start_time
                    rows_per_sec = processed_rows / elapsed
                    eta = (total_rows - processed_rows) / rows_per_sec
                    logging.info(f"Progress: {progress:.1f}% | Speed: {rows_per_sec:.0f} rows/s | ETA: {eta:.0f}s")

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {str(e)}")
                continue

        logging.info("Cleaning and normalizing data...")
        valid_rows = ~np.isnan(all_features).any(axis=1) & ~np.all(all_features == 0, axis=1)
        cleaned_data = all_features[valid_rows]

        # Log statistics about the data
        logging.info(f"Data statistics:")
        logging.info(f"  - Original rows: {total_rows:,}")
        logging.info(f"  - Valid rows: {len(cleaned_data):,}")
        logging.info(f"  - Invalid/filtered rows: {total_rows - len(cleaned_data):,}")

        # Handle outliers
        percentiles = np.percentile(cleaned_data, [1, 99], axis=0)
        for i in range(cleaned_data.shape[1]):
            cleaned_data[:, i] = np.clip(cleaned_data[:, i], percentiles[0, i], percentiles[1, i])

        mean = np.mean(cleaned_data, axis=0)
        std = np.std(cleaned_data, axis=0)
        logging.info(f"Feature statistics:")
        for i, (m, s) in enumerate(zip(mean, std)):
            logging.info(f"  Feature {i}: mean={m:.2f}, std={s:.2f}")

        normalized_data = (cleaned_data - mean) / (std + 1e-8)

        # Convert to tensor and save with weights_only=True
        tensor_data = torch.FloatTensor(normalized_data)
        logging.info(f"Saving cache file...")
        torch.save(tensor_data, self.cache_path, weights_only=True)

        cache_size = self.cache_path.stat().st_size / (1024 * 1024)
        logging.info(f"Cache file saved ({cache_size:.2f} MB)")

        total_time = time.time() - start_time
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        logging.info(f"Processing speed: {processed_rows / total_time:.0f} rows/s")

        return tensor_data


class TrafficPredictor(nn.Module):
    """Enhanced neural network model for traffic prediction"""

    def __init__(self, input_size=3, sequence_length=12, hidden_sizes=[128, 64], output_size=3, dropout_rate=0.2):
        super().__init__()

        self.sequence_length = sequence_length

        # Bidirectional LSTM for better sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True  # Enable bidirectional processing
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_sizes[0] * 2,  # *2 for bidirectional
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )

        # Enhanced fully connected layers with residual connections
        layers = []
        prev_size = hidden_sizes[0] * 2  # *2 for bidirectional

        for hidden_size in hidden_sizes[1:]:
            layers.extend([
                nn.LayerNorm(prev_size),  # Add layer normalization
                nn.Linear(prev_size, hidden_size),
                nn.GELU(),  # Use GELU activation
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer with layer normalization
        layers.extend([
            nn.LayerNorm(prev_size),
            nn.Linear(prev_size, output_size)
        ])

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(x)

        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling with residual connection
        pooled = torch.mean(attn_out + lstm_out, dim=1)

        # Final prediction
        return self.fc_layers(pooled)


class TrafficDataset(Dataset):
    """Dataset class for traffic data"""

    def __init__(self, data, sequence_length=12):
        """
        Initialize the dataset with preprocessed tensor data

        Args:
            data (torch.Tensor): Preprocessed and normalized data tensor
            sequence_length (int): Length of input sequences
        """
        self.sequence_length = sequence_length
        self.data = data
        logging.info(f"Initialized dataset with {len(data)} samples")

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Get sequence of data points
        sequence = self.data[idx:idx + self.sequence_length]
        # Get the next point as target
        target = self.data[idx + self.sequence_length]
        return sequence, target


class PrefetchDataLoader:
    """Custom dataloader with prefetching capability"""

    def __init__(self, dataloader, device, queue_size=3):
        self.dataloader = dataloader
        self.device = device
        self.queue = queue.Queue(maxsize=queue_size)
        self.iterator = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload_thread = Thread(target=self._preload_data, daemon=True)
        self.preload_thread.start()

    def _preload_data(self):
        try:
            while True:
                with torch.cuda.stream(self.stream):
                    try:
                        batch = next(self.iterator)
                        sequences, targets = batch
                        sequences = sequences.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)
                        self.queue.put((sequences, targets))
                    except StopIteration:
                        self.queue.put(None)
                        break
        except Exception as e:
            logging.error(f"Error in prefetch thread: {str(e)}")
            self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.queue.get()
        if batch is None:
            raise StopIteration
        return batch


class TrainingManager:
    """Highly optimized training manager with validation support"""

    def __init__(self, config, district):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.district = district
        self.checkpoint_dir = Path(config.config['PATHS']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Performance optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

        # Use lower precision for better speed
        self.scaler = torch.amp.GradScaler(
            init_scale=2 ** 10,
            growth_factor=1.5,
            backoff_factor=0.5,
            growth_interval=100
        )

        # Pre-allocate tensors for stats
        self.loss_tensor = torch.zeros(1, device=self.device)
        self.stats_cache = {}

        # Initialize logging
        self.log_buffer = []
        self.log_interval = 50
        self.flush_interval = 1000
        self.last_log_time = time.time()

    def _format_time(self, seconds):
        """Format time in a human-readable way"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def _is_bad_batch(self, sequences, targets):
        """Check for invalid batches"""
        return (
                torch.isnan(sequences).any() or
                torch.isnan(targets).any() or
                torch.isinf(sequences).any() or
                torch.isinf(targets).any()
        )

    def _training_step(self, model, sequences, targets, optimizer, criterion):
        """Execute one training step"""
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs = model(sequences)
            loss = criterion(outputs, targets)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        self.scaler.step(optimizer)
        self.scaler.update()

        return loss

    def _save_checkpoint_efficient(self, model, optimizer, epoch, loss):
        """Save model checkpoint efficiently"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        checkpoint_path = self.checkpoint_dir / f'district_{self.district}_checkpoint_{epoch + 1}.pt'
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
        logging.info(f"Saved checkpoint: {checkpoint_path}")

    def train_model(self, model, train_loader, val_loader=None):
        """Train the model with validation support"""
        model = model.to(self.device)
        model.train()

        # Optimize model for training
        if hasattr(model, 'fuse_model'):
            model.fuse_model()

        if torch.cuda.get_device_capability()[0] >= 7:
            model = model.cuda().half()

        criterion = nn.MSELoss(reduction='mean')

        # Use fused Adam if available
        use_fused = 'fused' in inspect.signature(optim.Adam).parameters
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(self.config.config['TRAINING']['learning_rate']),
            weight_decay=0.001,
            fused=use_fused
        )

        epochs = int(self.config.config['TRAINING']['epochs'])
        best_val_loss = float('inf')
        training_start = time.time()

        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_start = time.time()
            train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')

            train_loss = 0
            num_batches = 0

            for sequences, targets in train_progress:
                sequences = sequences.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if self._is_bad_batch(sequences, targets):
                    continue

                loss = self._training_step(model, sequences, targets, optimizer, criterion)
                train_loss += loss.item()
                num_batches += 1

                train_progress.set_postfix({'loss': f"{loss.item():.6f}"})

            avg_train_loss = train_loss / num_batches if num_batches > 0 else float('inf')

            # Validation phase
            if val_loader is not None:
                model.eval()
                val_loss = 0
                val_batches = 0
                val_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')

                with torch.no_grad():
                    for sequences, targets in val_progress:
                        sequences = sequences.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)

                        if self._is_bad_batch(sequences, targets):
                            continue

                        with torch.cuda.amp.autocast():
                            outputs = model(sequences)
                            loss = criterion(outputs, targets)

                        val_loss += loss.item()
                        val_batches += 1
                        val_progress.set_postfix({'val_loss': f"{loss.item():.6f}"})

                avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')

                # Save checkpoint if validation improves
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save_checkpoint_efficient(model, optimizer, epoch, avg_val_loss)
                    logging.info(f"New best validation loss: {avg_val_loss:.6f}")

            epoch_time = time.time() - epoch_start
            logging.info(
                f"\nEpoch {epoch + 1}/{epochs} completed in {self._format_time(epoch_time)}"
                f"\n  Training Loss: {avg_train_loss:.6f}"
                + (f"\n  Validation Loss: {avg_val_loss:.6f}" if val_loader else "")
            )

            # Clear memory periodically
            if epoch % 2 == 0:
                torch.cuda.empty_cache()

        total_time = time.time() - training_start
        logging.info(f"Training completed in {self._format_time(total_time)}")
        logging.info(f"Best validation loss: {best_val_loss:.6f}")


def main():
    try:
        # Initialize config and paths
        config = Config()
        base_path = verify_paths(config)
        model_dir = Path(config.config['PATHS']['model_dir'])
        model_dir.mkdir(exist_ok=True)

        # Set up district path
        district = 3
        district_path = base_path / f'district_{district}'

        # Initialize preprocessor and get data
        preprocessor = DataPreprocessor(district_path)
        processed_data = preprocessor.preprocess_and_cache()

        # Create dataset with proper train/val split
        total_size = len(processed_data)
        train_size = int(0.8 * total_size)
        train_data = processed_data[:train_size]
        val_data = processed_data[train_size:]

        train_dataset = TrafficDataset(train_data)
        val_dataset = TrafficDataset(val_data)

        # Optimized DataLoader configuration
        dataloader_config = {
            'batch_size': 512,  # Increased batch size
            'num_workers': min(8, os.cpu_count()),
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2,
            'drop_last': True
        }

        train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_config)
        val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_config)

        # Initialize enhanced model
        model = TrafficPredictor(
            input_size=3,
            sequence_length=12,
            hidden_sizes=[128, 64],  # Increased capacity
            output_size=3,
            dropout_rate=0.2
        )

        # Use model compilation if available
        if hasattr(torch, 'compile'):
            model = torch.compile(
                model,
                mode='max-autotune',
                fullgraph=True
            )

        # Initialize trainer with validation support
        trainer = TrainingManager(config, district)
        trainer.train_model(model, train_loader, val_loader)

        # Save final model with metadata
        final_model_path = model_dir / f'district_{district}_model.pt'
        metadata = {
            'model_state': model.state_dict(),
            'model_config': {
                'input_size': 3,
                'sequence_length': 12,
                'hidden_sizes': [128, 64],
                'output_size': 3,
                'dropout_rate': 0.2
            },
            'training_config': dataloader_config,
            'timestamp': time.strftime('%Y%m%d-%H%M%S')
        }
        torch.save(metadata, final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
