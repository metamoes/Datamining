import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import configparser
import logging
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()

        # Check if config file exists
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)

        self.config.read(config_path)

        # Load all configurations
        self.load_config()

        # Create necessary directories
        self.create_directories()

    def load_config(self):
        self.system = dict(self.config['SYSTEM'])
        self.model = dict(self.config['MODEL'])
        self.training = dict(self.config['TRAINING'])

        # Convert types
        self.batch_size = int(self.system['batch_size'])
        self.num_workers = int(self.system['num_workers'])

        # Auto-detect device
        if self.system['device'].lower() == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cuda':
                # Check for 2080 Super
                gpu_name = torch.cuda.get_device_name()
                if '2080' not in gpu_name:
                    logger.warning(f"Expected RTX 2080 Super, found: {gpu_name}")
        else:
            self.device = torch.device(self.system['device'])

    def create_directories(self):
        """Create all necessary directories for the project."""
        directories = [
            self.system['checkpoint_dir'],
            self.system['weights_dir'],
            self.system['cache_dir']
        ]

        for directory in directories:
            path = Path(directory)
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory created/verified: {path}")
            except Exception as e:
                logger.error(f"Failed to create directory {path}: {str(e)}")
                sys.exit(1)


class TrafficDataset(Dataset):
    def __init__(self, data_dir, district, cache_dir=None):
        self.data_dir = Path(data_dir)
        self.district = district
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Verify data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.cache_file = self.cache_dir / f"district_{district}_cache.pt" if self.cache_dir else None

        try:
            if self.cache_file and self.cache_file.exists():
                logger.info(f"Loading cached data for district {district}")
                self.data = torch.load(self.cache_file)
            else:
                logger.info(f"Processing data for district {district}")
                self.data = self._load_data()
                if self.cache_file:
                    logger.info(f"Caching data for district {district}")
                    torch.save(self.data, self.cache_file)
        except Exception as e:
            logger.error(f"Error processing district {district}: {str(e)}")
            raise

    def _load_data(self):
        district_path = self.data_dir / f'district_{self.district}'
        if not district_path.exists():
            raise FileNotFoundError(f"District directory not found: {district_path}")

        files = list(district_path.glob('*.txt'))
        if not files:
            raise FileNotFoundError(f"No data files found in {district_path}")

        data = []
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [executor.submit(self._process_file, f) for f in files]

            with tqdm(total=len(files), desc=f"Loading District {self.district}") as pbar:
                for future in futures:
                    try:
                        result = future.result()
                        if result:  # Only add if we got valid data
                            data.extend(result)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing file: {str(e)}")

        if not data:
            raise ValueError(f"No valid data loaded for district {self.district}")

        return torch.tensor(data, dtype=torch.float32)

    def _process_file(self, file_path):
        try:
            df = pd.read_csv(
                file_path,
                header=None,
                on_bad_lines='skip'  # Skip problematic lines
            )
            # Basic data validation
            if df.empty:
                logger.warning(f"Empty file: {file_path}")
                return None

            # Process the data according to the format specified
            # Extract relevant columns and convert to proper format
            processed_data = []
            for _, row in df.iterrows():
                try:
                    # Extract features from the row
                    features = [
                        float(x) for x in row.values[7:10]  # Samples, % Observed, Total Flow
                    ]
                    processed_data.append(features)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error processing row in {file_path}: {str(e)}")
                    continue

            return processed_data
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TrafficLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_size = int(config.model['input_features'])
        self.hidden_size = int(config.model['hidden_size'])
        self.num_layers = int(config.model['hidden_layers'])
        self.dropout = float(config.model['dropout_rate'])

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions


class TrafficPredictor:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.optimizers = {}

        # Initialize models for each district
        for district in range(1, 12):  # 1-11 inclusive
            try:
                self.models[district] = TrafficLSTM(config).to(config.device)
                self.optimizers[district] = optim.Adam(
                    self.models[district].parameters(),
                    lr=float(config.model['learning_rate'])
                )
                logger.info(f"Initialized model for district {district}")
            except Exception as e:
                logger.error(f"Error initializing model for district {district}: {str(e)}")
                raise

    def load_checkpoint(self, district):
        checkpoint_path = Path(self.config.system['checkpoint_dir']) / f"district_{district}_checkpoint.pt"
        try:
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                self.models[district].load_state_dict(checkpoint['model_state'])
                self.optimizers[district].load_state_dict(checkpoint['optimizer_state'])
                logger.info(f"Loaded checkpoint for district {district}")
                return checkpoint['epoch']
            return 0
        except Exception as e:
            logger.error(f"Error loading checkpoint for district {district}: {str(e)}")
            return 0

    def save_checkpoint(self, district, epoch):
        checkpoint_path = Path(self.config.system['checkpoint_dir']) / f"district_{district}_checkpoint.pt"
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state': self.models[district].state_dict(),
                'optimizer_state': self.optimizers[district].state_dict()
            }, checkpoint_path)
            logger.info(f"Saved checkpoint for district {district}")
        except Exception as e:
            logger.error(f"Error saving checkpoint for district {district}: {str(e)}")

    def train_district(self, district, train_loader, val_loader=None):
        model = self.models[district]
        optimizer = self.optimizers[district]
        criterion = nn.MSELoss()

        start_epoch = self.load_checkpoint(district)
        epochs = int(self.config.model['epochs'])

        try:
            for epoch in range(start_epoch, epochs):
                model.train()
                total_loss = 0

                with tqdm(train_loader, desc=f"District {district} Epoch {epoch}") as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        optimizer.zero_grad()

                        inputs = batch.to(self.config.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, inputs)

                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})

                if (epoch + 1) % int(self.config.training['checkpoint_frequency']) == 0:
                    self.save_checkpoint(district, epoch + 1)
        except Exception as e:
            logger.error(f"Error training district {district}: {str(e)}")
            raise


def main():
    try:
        logger.info("Initializing configuration...")
        config = Config()

        logger.info("Creating predictor...")
        predictor = TrafficPredictor(config)

        # Train each district
        for district in range(1, 12):
            try:
                logger.info(f"Processing district {district}...")
                dataset = TrafficDataset(
                    config.system['data_dir'],
                    district,
                    config.system['cache_dir']
                )

                loader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    shuffle=True,
                    pin_memory=True
                )

                predictor.train_district(district, loader)
            except Exception as e:
                logger.error(f"Error processing district {district}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
