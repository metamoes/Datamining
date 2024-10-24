# PeMS Data Downloader

An automated tool for downloading traffic data from California's Performance Measurement System (PeMS) website. This script handles authentication, navigation, and bulk downloading of station data while maintaining district organization and providing detailed logging.

## Features

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
- **Configurable Settings**:
  - Customizable download delays
  - Timeout settings
  - Output directory configuration

## Prerequisites

- Python 3.7+
- Chrome browser installed
- Valid PeMS account credentials

## Required Python Packages

```bash
pip install selenium
pip install requests
pip install urllib3
pip install webdriver-manager
pip install configparser
```

## Configuration

Create a `pems_config.ini` file in the script directory:

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
d7 = false
d8 = true
# Add more districts as needed
```

## Directory Structure

The script creates the following directory structure:
```
pems_data/
├── logs/
├── metadata/
├── processed_data/
├── raw_data/
│   ├── district_3/
│   ├── district_4/
│   └── district_8/
└── temp_downloads/
```

## Usage

1. Set up your configuration file:
```bash
# Copy the example config
cp example_pems_config.ini pems_config.ini
# Edit with your credentials and settings
nano pems_config.ini
```

2. Run the script:
```bash
python pems_downloader.py
```

## Logging and Monitoring

- Real-time progress is displayed in the console
- Detailed logs are stored in `pems_data/logs/download_log.txt`
- Download metadata is saved in `pems_data/metadata/download_metadata.json`
- Screenshots of key pages are saved for debugging purposes
