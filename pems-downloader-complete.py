import requests
import pandas as pd
import time 
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
import sys
import json
import configparser
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class PeMSDownloader:
    def __init__(self, config_path: str = "pems_config.ini"):
        """
        Initialize PeMS downloader using config file
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        # Set credentials
        self.username = self.config['credentials']['username']
        self.password = self.config['credentials']['password']
        
        # Set basic parameters
        self.base_url = "https://pems.dot.ca.gov"
        self.session = requests.Session()
        self.delay = float(self.config['download_settings'].get('delay_between_requests', 2))
        
        # Add common headers to mimic a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Set output directory
        self.output_dir = Path(self.config['download_settings'].get('output_dir', 'pems_data'))
        
        # Set dates
        end_date = self.config['download_settings'].get('end_date', '')
        if end_date:
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            self.end_date = datetime.now()
            
        start_date = self.config['download_settings'].get('start_date', '')
        if start_date:
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            self.start_date = self.end_date - timedelta(days=365)
        
        # Setup logging
        self.setup_logging()
        
        # Create output directory structure
        self.setup_directories()
        
        # Initialize metadata
        self.init_metadata()

    def init_metadata(self):
        """Initialize metadata structure"""
        self.metadata = {
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "districts": {},
            "stations": {},
            "download_stats": {
                "total_stations": 0,
                "successful_downloads": 0,
                "failed_downloads": 0
            }
        }

    def setup_logging(self):
        """Configure logging to both file and console"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'download_log.txt'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create directory structure for outputs"""
        dirs = ['raw_data', 'processed_data', 'metadata', 'logs']
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def login(self) -> bool:
        """Login to PeMS website with improved error handling"""
        try:
            # First get the login page to capture any necessary cookies or tokens
            login_page_url = f"{self.base_url}/"
            login_page = self.session.get(login_page_url, verify=False)
            if login_page.status_code != 200:
                self.logger.error(f"Failed to access login page. Status code: {login_page.status_code}")
                return False

            # Prepare login data
            login_url = f"{self.base_url}/index.php"
            payload = {
                'action': 'login',
                'username': self.username,
                'password': self.password,
                'redirect': '',
                'login': 'Login'
            }

            # Add additional headers for login request
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Origin': self.base_url,
                'Referer': f"{self.base_url}/"
            }

            # Attempt login
            response = self.session.post(login_url, data=payload, headers=headers, verify=False)
            
            # Debug logging
            self.logger.info(f"Login response status code: {response.status_code}")
            
            # Check if login was successful by making a test request to a protected page
            test_url = f"{self.base_url}/clearinghouse"
            test_response = self.session.get(test_url, verify=False)
            
            if test_response.status_code == 200 and 'logout' in test_response.text.lower():
                self.logger.info("Successfully logged in to PeMS")
                return True
            else:
                self.logger.error("Failed to verify login - protected page test failed")
                self.logger.debug(f"Test response content: {test_response.text[:500]}...")
                return False

        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error during login: {str(e)}")
            return False
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Timeout during login: {str(e)}")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error during login: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during login: {str(e)}")
            return False

    def get_enabled_districts(self) -> List[str]:
        """Get list of enabled districts from config"""
        districts = []
        for key, value in self.config['districts'].items():
            if value.lower() == 'true' and key.startswith('d'):
                districts.append(key[1:])  # Remove 'd' prefix
        return sorted(districts)

    def get_stations(self, district: str) -> List[Dict]:
        """Get list of stations for a district"""
        try:
            # Build URL for station list
            station_url = f"{self.base_url}/clearinghouse/district_stations.php?district_id={district}"
            response = self.session.get(station_url, verify=False)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to get stations for district {district}")
                return []

            # Parse HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            stations = []
            
            # Find the station table and parse rows
            # Note: You'll need to adjust these selectors based on actual HTML structure
            station_table = soup.find('table', {'id': 'stations'})
            if station_table:
                for row in station_table.find_all('tr')[1:]:  # Skip header row
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        station_info = {
                            'id': cols[0].text.strip(),
                            'name': cols[1].text.strip(),
                            'type': cols[2].text.strip()
                        }
                        stations.append(station_info)
            
            self.logger.info(f"Found {len(stations)} stations in district {district}")
            return stations

        except Exception as e:
            self.logger.error(f"Error getting stations for district {district}: {str(e)}")
            return []

    def download_station_data(self, district: str, station: Dict, current_date: datetime) -> bool:
        """Download data for a specific station and date"""
        try:
            # Format date for URL
            date_str = current_date.strftime("%Y%m%d")
            
            # Build download URL
            download_url = (f"{self.base_url}/clearinghouse/downloads/station/"
                          f"submit.php?district_id={district}&station_id={station['id']}"
                          f"&date={date_str}&type=raw")
            
            # Create output path
            output_path = (self.output_dir / 'raw_data' / district / 
                         station['id'] / f"{date_str}.txt.gz")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            response = self.session.get(download_url, stream=True, verify=False)
            if response.status_code == 200 and len(response.content) > 100:  # Basic size check
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Update metadata
                if district not in self.metadata['districts']:
                    self.metadata['districts'][district] = {
                        'total_stations': 0,
                        'processed_stations': 0
                    }
                
                if station['id'] not in self.metadata['stations']:
                    self.metadata['stations'][station['id']] = {
                        'name': station['name'],
                        'type': station['type'],
                        'district': district,
                        'downloads': {
                            'successful': 0,
                            'failed': 0
                        }
                    }
                
                self.metadata['stations'][station['id']]['downloads']['successful'] += 1
                self.metadata['download_stats']['successful_downloads'] += 1
                
                self.logger.info(f"Successfully downloaded data for station {station['id']} "
                               f"on {date_str}")
                return True
            
            else:
                if station['id'] in self.metadata['stations']:
                    self.metadata['stations'][station['id']]['downloads']['failed'] += 1
                self.metadata['download_stats']['failed_downloads'] += 1
                self.logger.error(f"Failed to download data for station {station['id']} "
                                f"on {date_str}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error downloading station {station['id']} data: {str(e)}")
            return False

    def save_metadata(self):
        """Save metadata to JSON file"""
        metadata_path = self.output_dir / 'metadata' / 'download_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def run(self):
        """Main execution function"""
        if not self.login():
            self.logger.error("Login failed. Exiting...")
            return
        
        # Get enabled districts from config
        districts = self.get_enabled_districts()
        
        for district in districts:
            self.logger.info(f"Processing district {district}")
            
            # Get stations for district
            stations = self.get_stations(district)
            if not stations:
                self.logger.warning(f"No stations found for district {district}")
                continue
                
            self.metadata['districts'][district] = {
                'total_stations': len(stations),
                'processed_stations': 0
            }
            self.metadata['download_stats']['total_stations'] += len(stations)
            
            for station in stations:
                self.logger.info(f"Processing station {station['id']}")
                
                # Process each day
                current_date = self.start_date
                while current_date <= self.end_date:
                    # Download data
                    success = self.download_station_data(district, station, current_date)
                    
                    # Add delay between downloads
                    time.sleep(self.delay)
                    
                    # Move to next day
                    current_date += timedelta(days=1)
                
                self.metadata['districts'][district]['processed_stations'] += 1
                
                # Save metadata after each station
                self.save_metadata()
                
            self.logger.info(f"Completed district {district}")
        
        self.logger.info("Download process complete")
        
        # Print final statistics
        total_downloads = (self.metadata['download_stats']['successful_downloads'] + 
                         self.metadata['download_stats']['failed_downloads'])
        success_rate = (self.metadata['download_stats']['successful_downloads'] / 
                       total_downloads * 100 if total_downloads > 0 else 0)
        
        self.logger.info("\nFinal Statistics:")
        self.logger.info(f"Total stations processed: {self.metadata['download_stats']['total_stations']}")
        self.logger.info(f"Successful downloads: {self.metadata['download_stats']['successful_downloads']}")
        self.logger.info(f"Failed downloads: {self.metadata['download_stats']['failed_downloads']}")
        self.logger.info(f"Success rate: {success_rate:.2f}%")

if __name__ == "__main__":
    try:
        downloader = PeMSDownloader("pems_config.ini")
        downloader.run()
    except Exception as e:
        print(f"Error during script execution: {str(e)}")
