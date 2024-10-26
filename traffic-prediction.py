import configparser
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
from torch.cuda.amp import autocast, GradScaler
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
    """Neural network model for traffic prediction"""

    def __init__(self, input_size=3, sequence_length=12, hidden_sizes=[64, 32], output_size=3, dropout_rate=0.2):
        super().__init__()

        self.sequence_length = sequence_length

        # LSTM layer to process the sequence
        self.lstm = nn.LSTM(
            input_size=input_size,  # 3 features per time step
            hidden_size=hidden_sizes[0],
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True
        )

        # Build fully connected layers for final prediction
        layers = []
        prev_size = hidden_sizes[0]

        for hidden_size in hidden_sizes[1:]:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)

        # Process sequence with LSTM
        lstm_out, _ = self.lstm(x)

        # Take the last time step's output
        lstm_last = lstm_out[:, -1, :]

        # Final prediction
        return self.fc_layers(lstm_last)


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


class TrafficPredictor(nn.Module):
    """Neural network model with enhanced performance"""

    def __init__(self, input_size=3, sequence_length=12, hidden_sizes=[64, 32], output_size=3, dropout_rate=0.2):
        super().__init__()

        self.sequence_length = sequence_length

        # Use faster LSTM implementation
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True,
            bias=True
        )

        # Optimize fully connected layers
        layers = []
        prev_size = hidden_sizes[0]

        for hidden_size in hidden_sizes[1:]:
            layers.extend([
                nn.Linear(prev_size, hidden_size, bias=True),
                nn.ReLU(inplace=True),  # Use inplace operations
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.fc_layers = nn.Sequential(*layers)

        # Initialize weights for faster convergence
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]
        return self.fc_layers(lstm_last)


class TrainingManager:
    """Highly optimized training manager with CUDA optimizations and memory efficiency"""

    def __init__(self, config, district):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.district = district
        self.checkpoint_dir = Path(config.config['PATHS']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Enable CUDA optimizations
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matrix multiplications
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cudnn

        # Initialize gradient scaler with dynamic scaling
        self.scaler = torch.amp.GradScaler(
            init_scale=2 ** 16,
            growth_factor=2,
            backoff_factor=0.5,
            growth_interval=2000
        )

        # Pre-allocate pinned memory for faster CPU->GPU transfers
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()

        logging.info(f"Initialized TrainingManager with optimizations:")
        logging.info(f"  - Device: {self.device}")
        logging.info(f"  - Checkpoint directory: {self.checkpoint_dir}")
        logging.info(f"  - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        logging.info(f"  - CUDA memory cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    def train_model(self, model, train_loader):
        model = model.to(self.device)
        model.train()

        # Enable tensor cores if available
        if torch.cuda.get_device_capability()[0] >= 7:
            logging.info("Tensor cores enabled for compatible operations")

        # Log model architecture and parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model architecture:")
        logging.info(f"  - Total parameters: {total_params:,}")
        logging.info(f"  - Trainable parameters: {trainable_params:,}")

        # Optimize memory layout
        try:
            model = torch.jit.script(model)  # JIT compilation for faster execution
        except Exception as e:
            logging.warning(f"JIT compilation failed, using standard model: {str(e)}")

        # Use scaled loss for better numerical stability
        criterion = nn.MSELoss(reduction='mean')

        # Configure optimizer with momentum and nesterov
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(self.config.config['TRAINING']['learning_rate']),
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True  # Enable AMSGrad variant
        )

        # Performance optimized scheduler
        total_steps = int(self.config.config['TRAINING']['epochs']) * len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(self.config.config['TRAINING']['learning_rate']),
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95
        )

        prefetch_loader = PrefetchDataLoader(train_loader, self.device, queue_size=3)
        total_start_time = time.time()
        best_loss = float('inf')
        running_loss = 0.0

        # Initialize CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for epoch in range(int(self.config.config['TRAINING']['epochs'])):
            epoch_start_time = time.time()
            start_event.record()

            progress_bar = tqdm(
                prefetch_loader,
                desc=f'Epoch {epoch + 1}/{self.config.config["TRAINING"]["epochs"]}',
                dynamic_ncols=True  # Adapt to terminal width
            )

            batch_times = []
            for batch_idx, (sequences, targets) in enumerate(progress_bar):
                batch_start = torch.cuda.Event(enable_timing=True)
                batch_end = torch.cuda.Event(enable_timing=True)
                batch_start.record()

                # Zero gradients with optimized memory handling
                for param in model.parameters():
                    param.grad = None

                # Compute with automatic mixed precision
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(sequences)
                    loss = criterion(outputs, targets)

                # Scale loss and backward pass
                self.scaler.scale(loss).backward()

                # Unscale gradients and clip for stability
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update weights with gradient scaling
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()

                # Efficient loss computation
                running_loss = 0.99 * running_loss + 0.01 * loss.item()

                # Calculate batch time
                batch_end.record()
                torch.cuda.synchronize()
                batch_time = batch_start.elapsed_time(batch_end) / 1000  # Convert to seconds
                batch_times.append(batch_time)

                # Update progress less frequently
                if batch_idx % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    avg_batch_time = np.mean(batch_times[-100:]) if batch_times else 0
                    progress_bar.set_postfix({
                        'loss': f"{running_loss:.6f}",
                        'lr': f"{current_lr:.6f}",
                        'batch_time': f"{avg_batch_time:.3f}s",
                        'memory': f"{torch.cuda.memory_allocated() / 1024 ** 2:.0f}MB"
                    })

            # Record epoch end time
            end_event.record()
            torch.cuda.synchronize()
            epoch_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds

            # Calculate epoch statistics
            epoch_loss = running_loss

            # Log epoch statistics
            logging.info(f"Epoch {epoch + 1} completed:")
            logging.info(f"  - Loss: {epoch_loss:.6f}")
            logging.info(f"  - Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            logging.info(f"  - Time: {epoch_time:.2f}s")
            logging.info(f"  - Average batch time: {np.mean(batch_times):.3f}s")
            logging.info(f"  - CUDA memory: {torch.cuda.memory_allocated() / 1024 ** 2:.0f}MB")

            # Save best model with async I/O
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
                self._save_checkpoint(model, optimizer, epoch, epoch_loss)
                logging.info(f"  - New best loss achieved!")

            # Periodic checkpoints
            if (epoch + 1) % int(self.config.config['TRAINING']['checkpoint_frequency']) == 0:
                self._save_checkpoint(model, optimizer, epoch, epoch_loss)

            # Clear cache periodically
            if (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()

        total_time = time.time() - total_start_time
        logging.info(f"Training completed:")
        logging.info(f"  - Total time: {total_time:.2f}s")
        logging.info(f"  - Best loss: {best_loss:.6f}")
        logging.info(f"  - Final CUDA memory: {torch.cuda.memory_allocated() / 1024 ** 2:.0f}MB")

    def _save_checkpoint(self, model, optimizer, epoch, loss):
        checkpoint_path = self.checkpoint_dir / f'district_{self.district}_checkpoint_{epoch + 1}.pt'

        # Create checkpoint with minimal memory overhead
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        # Async save with error handling
        try:
            torch.save(
                checkpoint,
                checkpoint_path,
                _use_new_zipfile_serialization=True,  # Use new format for faster saving
            )
            checkpoint_size = checkpoint_path.stat().st_size / (1024 * 1024)
            logging.info(f"Saved checkpoint {epoch + 1}:")
            logging.info(f"  - Path: {checkpoint_path}")
            logging.info(f"  - Size: {checkpoint_size:.2f} MB")
            logging.info(f"  - Loss: {loss:.6f}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {str(e)}")


def main():
    try:
        config = Config()
        base_path = verify_paths(config)
        model_dir = Path(config.config['PATHS']['model_dir'])
        model_dir.mkdir(exist_ok=True)

        district = 3
        district_path = base_path / f'district_{district}'

        # Use enhanced data preprocessing to get tensor data
        preprocessor = DataPreprocessor(district_path)
        processed_data = preprocessor.preprocess_and_cache()

        # Create dataset with the preprocessed tensor data
        dataset = TrafficDataset(processed_data)
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Initialize model
        model = TrafficPredictor(
            input_size=3,
            sequence_length=12,
            hidden_sizes=[64, 32],
            output_size=3,
            dropout_rate=0.2
        )

        # Train with enhanced manager
        trainer = TrainingManager(config, district)
        trainer.train_model(model, dataloader)

        # Save final model
        torch.save(model.state_dict(), model_dir / f'district_{district}_model.pt')

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
        raise



if __name__ == "__main__":
    main()
