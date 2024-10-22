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
        """Login to PeMS website matching the exact form structure"""
        try:
            # First get the main page to establish session and get any cookies
            main_url = "https://pems.dot.ca.gov"

            self.logger.info("Accessing main page...")
            main_response = self.session.get(main_url, verify=False)
            if main_response.status_code != 200:
                self.logger.error(f"Failed to access main page. Status: {main_response.status_code}")
                return False

            # The login form submits to the root URL with POST
            login_url = "https://pems.dot.ca.gov/"

            # Exactly match the form data from the HTML
            login_payload = {
                'redirect': '',  # Hidden input field
                'username': self.username,
                'password': self.password,
                'login': 'Login'  # Submit button value
            }

            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Origin': 'https://pems.dot.ca.gov',
                'Referer': 'https://pems.dot.ca.gov/',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Cache-Control': 'max-age=0',
                'Connection': 'keep-alive',
                'Host': 'pems.dot.ca.gov',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'same-origin',
                'Sec-Fetch-User': '?1',
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
            }

            self.logger.info(f"Attempting login with URL: {login_url}")
            login_response = self.session.post(
                login_url,
                data=login_payload,
                headers=headers,
                verify=False,
                allow_redirects=True
            )

            self.logger.info(f"Login response status: {login_response.status_code}")
            self.logger.info(f"Login response URL: {login_response.url}")

            # Check cookies
            self.logger.info("Current cookies:")
            for cookie in self.session.cookies:
                self.logger.info(f"Cookie {cookie.name}: {cookie.value}")

            # Verify login by checking a profile or clearinghouse page
            test_url = "https://pems.dot.ca.gov/?dnode=profile"
            self.logger.info(f"Testing authentication with URL: {test_url}")

            test_response = self.session.get(test_url, verify=False)
            self.logger.info(f"Test response status: {test_response.status_code}")
            self.logger.info(f"Test response URL: {test_response.url}")

            # If we're redirected back to the login page, login failed
            if 'login' in test_response.url.lower():
                self.logger.error("Login failed - redirected back to login page")
                return False

            # If we see profile or clearinghouse content, login succeeded
            if test_response.status_code == 200:
                self.logger.info("Successfully logged in to PeMS")
                return True

            self.logger.error("Login verification failed")
            return False

        except Exception as e:
            self.logger.error(f"Login error: {str(e)}")
            self.logger.error(f"Full error details: {str(e.__class__.__name__)}: {str(e)}")
            return False


    def get_enabled_districts(self) -> List[str]:
        """Get list of enabled districts from config"""
        districts = []
        for key, value in self.config['districts'].items():
            if value.lower() == 'true' and key.startswith('d'):
                districts.append(key[1:])  # Remove 'd' prefix
        return sorted(districts)

    def get_stations(self, district: str) -> List[Dict]:
        """Get list of stations by accessing the station_5min data type"""
        try:
            # First access the clearinghouse page with proper parameters
            url = "https://pems.dot.ca.gov/"
            params = {
                'dnode': 'Clearinghouse',
                'type': 'station_5min',
                'district_id': district,
                'submit': 'Submit'
            }

            self.logger.info(f"Accessing clearinghouse with params: {params}")
            response = self.session.get(url, params=params, verify=False)

            if response.status_code != 200:
                self.logger.error(f"Failed to access clearinghouse. Status: {response.status_code}")
                return []

            # Parse HTML for available stations
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get the data files table
            data_table = soup.find('table', {'id': 'datafiles'})
            if not data_table:
                self.logger.error("Could not find data files table")
                return []

            stations = []
            for row in data_table.find_all('tr')[1:]:  # Skip header
                cols = row.find_all('td')
                if len(cols) >= 2:
                    filename = cols[0].find('a')['href']
                    file_id = filename.split('download=')[1].split('&')[0]

                    station_info = {
                        'id': f"station_{file_id}",
                        'name': filename,
                        'type': 'station'
                    }
                    stations.append(station_info)

            self.logger.info(f"Found {len(stations)} files for district {district}")
            return stations

        except Exception as e:
            self.logger.error(f"Error getting stations for district {district}: {str(e)}")
            self.logger.error(f"Full error details: {str(e.__class__.__name__)}: {str(e)}")
            return []

    def download_station_data(self, district: str, station: Dict, current_date: datetime) -> bool:
        """Download station data file directly from clearinghouse"""
        try:
            # Format date for filename
            date_str = current_date.strftime("%Y_%m_%d")

            # Build download URL
            download_url = f"https://pems.dot.ca.gov/?download={station['id'].split('_')[1]}&dnode=Clearinghouse"

            # Create output path using district/date structure
            output_path = (self.output_dir / 'raw_data' / district /
                           date_str / f"{station['name']}")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file with progress logging
            self.logger.info(f"Downloading {download_url} to {output_path}")

            response = self.session.get(download_url, stream=True, verify=False)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                self.metadata['download_stats']['successful_downloads'] += 1
                self.logger.info(f"Successfully downloaded {station['name']}")
                return True

            else:
                self.metadata['download_stats']['failed_downloads'] += 1
                self.logger.error(f"Failed to download {station['name']}. Status: {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"Error downloading station data: {str(e)}")
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
