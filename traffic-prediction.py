import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import configparser
from tqdm import tqdm
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class TrafficDataset(Dataset):
    def __init__(self, data_path, sequence_length=12):
        self.sequence_length = sequence_length
        self.features = []
        self.targets = []
        
        # Process all files in directory
        for file_path in glob.glob(os.path.join(data_path, "*.txt")):
            self._process_file(file_path)
            
    def _process_file(self, file_path):
        df = pd.read_csv(file_path, header=None)
        # Extract relevant features (flow, speed, occupancy)
        # Assuming standard PeMS format
        features = []
        for row in df.values:
            # Parse timestamp
            timestamp = datetime.strptime(row[0], '%m/%d/%Y %H:%M:%S')
            # Extract flow, speed, occupancy for each lane
            lane_data = []
            for i in range(6, len(row), 5):  # Each lane has 5 values
                if pd.notna(row[i]) and pd.notna(row[i+1]) and pd.notna(row[i+2]):
                    lane_data.extend([float(row[i]), float(row[i+1]), float(row[i+2])])
            
            if lane_data:  # Only add if we have valid data
                features.append(lane_data)
        
        # Create sequences
        for i in range(len(features) - self.sequence_length):
            self.features.append(features[i:i+self.sequence_length])
            self.targets.append(features[i+self.sequence_length])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.targets[idx])

class TrafficPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TrafficPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class ModelTrainer:
    def __init__(self, config_path='config.ini'):
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        
    def _load_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def _init_model(self, district):
        model = TrafficPredictor(
            input_size=int(self.config['MODEL']['input_size']),
            hidden_size=int(self.config['MODEL']['hidden_size']),
            num_layers=int(self.config['MODEL']['num_layers'])
        ).to(self.device)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(self.config['TRAINING']['learning_rate'])
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=int(self.config['TRAINING']['scheduler_patience'])
        )
        
        return model, optimizer, scheduler

    def save_checkpoint(self, district, epoch, model, optimizer, scheduler, loss):
        checkpoint_path = os.path.join(
            self.config['PATHS']['checkpoint_dir'],
            f'model_{district}_checkpoint.pt'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }, checkpoint_path)

    def load_checkpoint(self, district):
        checkpoint_path = os.path.join(
            self.config['PATHS']['checkpoint_dir'],
            f'model_{district}_checkpoint.pt'
        )
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model, optimizer, scheduler = self._init_model(district)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            return model, optimizer, scheduler, checkpoint['epoch']
        
        return None

    def train_district(self, district, train_loader, val_loader=None):
        logging.info(f"Training model for district {district}")
        
        # Check for existing checkpoint
        checkpoint_data = self.load_checkpoint(district)
        if checkpoint_data:
            model, optimizer, scheduler, start_epoch = checkpoint_data
            logging.info(f"Resuming training from epoch {start_epoch}")
        else:
            model, optimizer, scheduler = self._init_model(district)
            start_epoch = 0

        criterion = nn.MSELoss()
        num_epochs = int(self.config['TRAINING']['num_epochs'])
        
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_features, batch_targets in pbar:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': total_loss/len(train_loader)})

            # Validation
            if val_loader:
                val_loss = self.validate(model, val_loader, criterion)
                scheduler.step(val_loss)
                
            # Save checkpoint
            self.save_checkpoint(district, epoch + 1, model, optimizer, scheduler, total_loss/len(train_loader))
            
        return model

    def validate(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

    def train_all_districts(self):
        base_path = self.config['PATHS']['data_dir']
        districts = [d for d in os.listdir(base_path) if d.startswith('district_')]
        
        for district in districts:
            district_path = os.path.join(base_path, district)
            dataset = TrafficDataset(district_path)
            
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(self.config['TRAINING']['batch_size']),
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=int(self.config['TRAINING']['batch_size'])
            )
            
            model = self.train_district(district, train_loader, val_loader)
            self.models[district] = model

        # Train combined model
        self._train_combined_model()

    def _train_combined_model(self):
        # Implementation for combined model training
        pass

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_districts()
