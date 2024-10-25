import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import configparser
import tqdm
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Config:
    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        
        # Default configurations
        self.config['DATA'] = {
            'data_path': 'E:/Dataset/pems_data/raw_data',
            'sample_data_path': 'sample_data',
            'sequence_length': '12',
            'prediction_horizon': '6',
            'batch_size': '32'
        }
        
        self.config['MODEL'] = {
            'hidden_size': '64',
            'num_layers': '2',
            'dropout': '0.2',
            'learning_rate': '0.001',
            'epochs': '100'
        }
        
        # Load existing config if it exists
        if os.path.exists(config_file):
            self.config.read(config_file)
        else:
            with open(config_file, 'w') as f:
                self.config.write(f)
    
    def get(self, section, key):
        return self.config[section][key]

class PEMSDataset(Dataset):
    def __init__(self, data_path, sequence_length=12):
        self.sequence_length = sequence_length
        self.data = self._load_and_preprocess_data(data_path)
        self.scaler = MinMaxScaler()
        self.data_scaled = self.scaler.fit_transform(self.data)
        
    def _load_and_preprocess_data(self, data_path):
        all_files = glob.glob(os.path.join(data_path, "district_*/*.txt"))
        dfs = []
        
        for file in tqdm.tqdm(all_files, desc="Loading data files"):
            df = self._process_single_file(file)
            if df is not None:
                dfs.append(df)
        
        return pd.concat(dfs, axis=0)
    
    def _process_single_file(self, file_path):
        try:
            # Read the raw data
            df = pd.read_csv(file_path, header=None)
            
            # Extract timestamp and traffic metrics
            df[0] = pd.to_datetime(df[0])
            
            # Extract relevant columns (flow, speed, occupancy for each lane)
            # Assuming columns follow the pattern described in the sample data
            traffic_cols = []
            for lane in range(8):  # Up to 8 lanes
                base_idx = 6 + lane * 5  # Starting index for each lane's metrics
                if base_idx + 3 < len(df.columns):
                    traffic_cols.extend([base_idx, base_idx + 1, base_idx + 2])
            
            return df[traffic_cols]
        
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def __len__(self):
        return len(self.data_scaled) - self.sequence_length
    
    def __getitem__(self, idx):
        sequence = self.data_scaled[idx:idx + self.sequence_length]
        target = self.data_scaled[idx + self.sequence_length]
        return torch.FloatTensor(sequence), torch.FloatTensor(target)

class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.linear = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class TrafficPredictor:
    def __init__(self, config_path='config.ini'):
        self.config = Config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        
    def setup_model(self):
        # Create data loaders
        self.dataset = PEMSDataset(
            self.config.get('DATA', 'data_path'),
            int(self.config.get('DATA', 'sequence_length'))
        )
        
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, test_size]
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(self.config.get('DATA', 'batch_size')),
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=int(self.config.get('DATA', 'batch_size')),
            shuffle=False
        )
        
        # Create model
        input_size = self.dataset.data.shape[1]
        self.model = TrafficLSTM(
            input_size=input_size,
            hidden_size=int(self.config.get('MODEL', 'hidden_size')),
            num_layers=int(self.config.get('MODEL', 'num_layers')),
            dropout=float(self.config.get('MODEL', 'dropout'))
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.get('MODEL', 'learning_rate'))
        )
    
    def train(self):
        epochs = int(self.config.get('MODEL', 'epochs'))
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            progress_bar = tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx, (sequences, targets) in enumerate(progress_bar):
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({'loss': train_loss/(batch_idx+1)})
            
            # Validation
            val_loss = self.evaluate()
            logging.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(self.train_loader):.4f} - Val Loss: {val_loss:.4f}')
            
            # Save checkpoint if best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint('best_model.pth')
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for sequences, targets in self.test_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(self.test_loader)
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.dataset.scaler
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dataset.scaler = checkpoint['scaler']

def main():
    predictor = TrafficPredictor()
    predictor.train()

if __name__ == "__main__":
    main()
