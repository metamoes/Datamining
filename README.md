# PeMS Traffic Analysis System
A comprehensive toolkit for downloading, processing, and analyzing traffic data from California's Performance Measurement System (PeMS). This system handles the complete pipeline from data acquisition through analysis and visualization.

## System Components
1. **Data Acquisition Module**: Automated downloading from PeMS website
2. **Data Processing Pipeline**: Comprehensive traffic data processing
3. **Neural Network Analysis**: Advanced traffic prediction system
4. **Visualization Suite**: Detailed traffic pattern visualization

## Features

### Data Acquisition
- **Automated Authentication**: Secure login to PeMS website with credential management
- **District-based Downloads**: Configure and download data for specific California districts
- **Smart File Management**: 
  - Maintains original file structure
  - Skips existing downloads
  - Organizes downloads by district
- **Robust Error Handling**:
  - Automatic retries for failed downloads
  - Comprehensive logging system
  - Session management with automatic recovery
- **Progress Tracking**:
  - Real-time download progress reporting
  - Download statistics and success rates
  - Detailed metadata generation

### Traffic Analysis
- **Advanced Traffic Engineering Metrics**:
  - Level of Service (LOS) calculations
  - Queue length estimation
  - Travel Time Index
  - Buffer Index for reliability
  - Wave speed analysis
  - Peak Hour Factor
  - Capacity utilization

- **Neural Network Capabilities**:
  - LSTM-based prediction
  - Attention mechanisms
  - Comprehensive cross-validation
  - District-specific models

- **Visualization Suite**:
  - Fundamental diagrams
  - Congestion patterns
  - Reliability metrics
  - Wave speed analysis
  - Performance heat maps

## Prerequisites
- Python 3.7 - 3.11.9
- Chrome browser installed (for data downloading)
- Valid PeMS account credentials
- CUDA-capable GPU (recommended for neural network training)

## Installation

### Single-line Installation Command
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm rich pyyaml configparser logging pathlib datetime typing tensorboard optuna pytest black pylint mypy selenium requests urllib3 webdriver-manager configparser
```

### Alternative: Using requirements.txt
```bash
pip install -r requirements.txt
```

## Directory Structure
```
pems_analysis/
├── data/
│   ├── raw_data/
│   │   ├── district_3/
│   │   ├── district_4/
│   │   └── ...
│   ├── processed_data/
│   └── metadata/
├── logs/
│   ├── download_logs/
│   ├── processing_logs/
│   └── training_logs/
├── models/
│   ├── checkpoints/
│   └── configs/
├── output/
│   ├── visualizations/
│   ├── reports/
│   └── predictions/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── models/
│   └── visualization/
└── tests/
```

## Configuration

### Data Acquisition Config (pems_config.ini)
```ini
[credentials]
username = your_pems_username
password = your_pems_password

[download_settings]
delay_between_requests = 2
timeout = 60
output_dir = pems_data

[districts]
d3 = true
d4 = true
d5 = false
d6 = true
```

### Model Config (model_config.ini)
```ini
[NetworkArchitecture]
input_size = 20
hidden_size = 128
num_layers = 2
dropout = 0.2

[Training]
batch_size = 64
epochs = 100
learning_rate = 0.001
```

## Usage

### 1. Data Download
```bash
python src/data_acquisition/pems_downloader.py
```

### 2. Data Processing
```bash
python src/preprocessing/process_data.py
```

### 3. Model Training
```bash
python src/models/train_model.py
```

### 4. Generate Visualizations
```bash
python src/visualization/generate_plots.py
```
