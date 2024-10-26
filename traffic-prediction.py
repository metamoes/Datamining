import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import configparser
import logging
from tqdm import tqdm
import os

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


class TrafficDataset(Dataset):
    """Dataset class for traffic data"""

    def __init__(self, district_path, sequence_length=12):
        self.sequence_length = sequence_length
        self.data = self._load_district_data(district_path)

    def _load_district_data(self, district_path):
        """Load and combine all data files for a district"""
        district_path = Path(district_path)
        logging.info(f"Searching for data files in: {district_path}")

        # Print all files in directory for debugging
        logging.info("Directory contents:")
        for item in district_path.glob('*'):
            logging.info(f"Found: {item}")

        # Try different file patterns
        patterns = [
            'd*_text_station_5min_*.txt',
            'district_*_*.txt',
            '*.txt'
        ]

        data_files = []
        for pattern in patterns:
            files = list(district_path.glob(pattern))
            if files:
                logging.info(f"Found {len(files)} files with pattern: {pattern}")
                data_files.extend(files)
                break

        if not data_files:
            logging.error(f"No data files found in {district_path} with any pattern")
            raise FileNotFoundError(f"No data files found in {district_path}")

        logging.info(f"Found {len(data_files)} total data files")

        all_data = []
        for file_path in tqdm(data_files, desc="Loading data files"):
            try:
                # Try to read the first few lines of the file for debugging
                with open(file_path, 'r') as f:
                    first_lines = [next(f) for _ in range(5)]
                logging.info(f"First few lines of {file_path.name}:")
                for line in first_lines:
                    logging.info(line.strip())

                # Read the file
                df = pd.read_csv(file_path, header=None)

                # Extract features (Total Flow, Avg Occupancy, Avg Speed)
                features = df.iloc[:, [8, 9, 10]].values

                # Handle missing or invalid values
                features = np.nan_to_num(features, nan=0.0)

                # Filter out invalid data
                valid_rows = ~np.all(features == 0, axis=1)
                features = features[valid_rows]

                if len(features) > 0:
                    all_data.append(features)
                    logging.info(f"Successfully loaded {len(features)} rows from {file_path.name}")

            except Exception as e:
                logging.error(f"Error loading file {file_path}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

        if not all_data:
            raise ValueError(f"No valid data could be loaded from {district_path}")

        # Combine all data
        combined_data = np.vstack(all_data)

        # Normalize the data
        mean = np.mean(combined_data, axis=0)
        std = np.std(combined_data, axis=0)
        normalized_data = (combined_data - mean) / (std + 1e-8)

        return torch.FloatTensor(normalized_data)

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]
        return sequence, target

class TrafficPredictor(nn.Module):
    """Neural network model for traffic prediction"""

    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super().__init__()

        # Convert hidden_sizes string to list of integers
        if isinstance(hidden_sizes, str):
            hidden_sizes = [int(size) for size in hidden_sizes.split(',')]

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TrainingManager:
    """Manages the training process and checkpoints"""

    def __init__(self, config, district):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.district = district
        self.checkpoint_dir = Path(config.config['PATHS']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Verify CUDA device
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            if '2080' not in gpu_name:
                logging.warning(f"Expected RTX 2080 Super, found: {gpu_name}")
        else:
            logging.warning("CUDA device not available, using CPU")

    def save_checkpoint(self, model, optimizer, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        path = self.checkpoint_dir / f'district_{self.district}_checkpoint_{epoch}.pt'
        torch.save(checkpoint, path)

    def load_checkpoint(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

    def train_model(self, model, train_loader):
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=float(self.config.config['TRAINING']['learning_rate']))

        # Load latest checkpoint if exists
        start_epoch = 0
        checkpoint_files = list(self.checkpoint_dir.glob(f'district_{self.district}_checkpoint_*.pt'))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
            start_epoch, _ = self.load_checkpoint(model, optimizer, latest_checkpoint)
            logging.info(f"Resuming training from epoch {start_epoch}")

        epochs = int(self.config.config['TRAINING']['epochs'])
        checkpoint_freq = int(self.config.config['TRAINING']['checkpoint_frequency'])

        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

            for batch_idx, (sequences, targets) in enumerate(progress_bar):
                sequences, targets = sequences.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})

            avg_loss = total_loss / len(train_loader)
            logging.info(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

            if (epoch + 1) % checkpoint_freq == 0:
                self.save_checkpoint(model, optimizer, epoch + 1, avg_loss)


def main():
    try:
        # Initialize configuration
        config = Config()

        # Verify paths
        base_path = verify_paths(config)

        # Create model directories
        model_dir = Path(config.config['PATHS']['model_dir'])
        model_dir.mkdir(exist_ok=True)

        # Initialize CUDA device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logging.warning("CUDA device not available, using CPU")

        # Process district 3
        district = 3
        district_path = base_path / f'district_{district}'

        if not district_path.exists():
            logging.error(f"District directory not found: {district_path}")
            return

        logging.info(f"Processing District {district}")
        logging.info(f"Looking for data in: {district_path}")

        # Create dataset
        try:
            dataset = TrafficDataset(district_path)
            dataloader = DataLoader(
                dataset,
                batch_size=int(config.config['TRAINING']['batch_size']),
                shuffle=True,
                num_workers=0
            )

            logging.info(f"Successfully created dataset with {len(dataset)} samples")

            # Initialize and train model
            model = TrafficPredictor(
                input_size=int(config.config['MODEL']['input_size']),
                hidden_sizes=config.config['MODEL']['hidden_layers'],
                output_size=int(config.config['MODEL']['output_size']),
                dropout_rate=float(config.config['MODEL']['dropout'])
            ).to(device)

            trainer = TrainingManager(config, district)
            trainer.train_model(model, dataloader)

            # Save the model
            torch.save(model.state_dict(), model_dir / f'district_{district}_model.pt')

        except Exception as e:
            logging.error(f"Error processing district {district}: {str(e)}")
            logging.error(traceback.format_exc())

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
