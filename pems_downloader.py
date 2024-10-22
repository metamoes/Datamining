import re
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

        # Setup logging
        self.setup_logging()

        # Create output directory structure
        self.setup_directories()

        # Initialize metadata
        self.init_metadata()

    def init_metadata(self):
        """Initialize metadata structure"""
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "districts": {},
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

    def save_metadata(self):
        """Save metadata to file"""
        metadata_path = self.output_dir / 'metadata' / 'download_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def login(self) -> bool:
        """Login to PeMS website"""
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

            # Verify login by checking a profile page
            test_url = "https://pems.dot.ca.gov/?dnode=profile"
            self.logger.info(f"Testing authentication with URL: {test_url}")

            test_response = self.session.get(test_url, verify=False)
            self.logger.info(f"Test response status: {test_response.status_code}")
            self.logger.info(f"Test response URL: {test_response.url}")

            # If we're redirected back to the login page, login failed
            if 'login' in test_response.url.lower():
                self.logger.error("Login failed - redirected back to login page")
                return False

            # If we see profile content, login succeeded
            if test_response.status_code == 200:
                self.logger.info("Successfully logged in to PeMS")
                return True

            self.logger.error("Login verification failed")
            return False

        except Exception as e:
            self.logger.error(f"Login error: {str(e)}")
            return False

    def get_enabled_districts(self) -> List[str]:
        """Get list of enabled districts from config"""
        districts = []
        for key, value in self.config['districts'].items():
            if value.lower() == 'true' and key.startswith('d'):
                districts.append(key[1:])  # Remove 'd' prefix
        return sorted(districts)

    def get_stations_with_retry(self, district: str, max_retries: int = 3, delay: float = 2.0) -> List[Dict]:
        """Get list of stations from the datafiles table"""
        for attempt in range(max_retries):
            try:
                # Initial clearinghouse page request
                clearinghouse_url = "https://pems.dot.ca.gov/"
                params = {
                    'dnode': 'Clearinghouse',
                    'type': 'station_5min',
                    'district_id': district,
                    'submit': 'Submit'
                }

                self.logger.info(f"Attempt {attempt + 1}: Accessing clearinghouse page...")
                response = self.session.get(clearinghouse_url, params=params, verify=False)

                if response.status_code != 200:
                    self.logger.error(f"Failed to access clearinghouse. Status: {response.status_code}")
                    continue

                # Parse the page
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find the datafiles table - it's in a div with class 'dataFilesScrollBox'
                scroll_box = soup.find('div', {'class': 'dataFilesScrollBox'})
                if not scroll_box:
                    self.logger.warning(f"Data files scroll box not found in attempt {attempt + 1}")
                    continue

                # Find the table with id 'datafiles'
                data_table = scroll_box.find('table', {'id': 'datafiles'})
                if not data_table:
                    self.logger.warning(f"Data files table not found in attempt {attempt + 1}")
                    continue

                # Parse file links from the table rows
                data_files = []
                rows = data_table.find_all('tr')

                for row in rows[1:]:  # Skip header row
                    try:
                        link = row.find('a')
                        if link and 'href' in link.attrs:
                            href = link['href']
                            file_name = link.text.strip()

                            # Extract download ID from href
                            download_match = re.search(r'download=(\d+)', href)
                            if download_match:
                                file_id = download_match.group(1)

                                # Only process files matching the expected pattern for 5-minute data
                                if re.match(rf'd{district}_text_station_5min_\d{{4}}_\d{{2}}_\d{{2}}\.txt\.gz',
                                            file_name):
                                    data_files.append({
                                        'id': f"station_{file_id}",
                                        'name': file_name,
                                        'type': 'station',
                                        'url': href
                                    })
                    except Exception as e:
                        self.logger.error(f"Error processing row: {str(e)}")
                        continue

                if data_files:
                    self.logger.info(f"Successfully found {len(data_files)} files for district {district}")
                    return data_files

                self.logger.warning(f"No files found in attempt {attempt + 1}, retrying...")
                time.sleep(delay)

            except Exception as e:
                self.logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                continue

        self.logger.error(f"Failed to get stations after {max_retries} attempts")
        return []

    def download_station_data(self, district: str, station: Dict) -> bool:
        """Download station data with improved error handling"""
        try:
            # Get the complete download URL
            download_url = f"https://pems.dot.ca.gov/{station['url'].lstrip('/')}" if station['url'].startswith(
                '/') else station['url']
            if not download_url.startswith('https://'):
                download_url = f"https://pems.dot.ca.gov/{download_url.lstrip('?')}"

            # Create output path using the actual filename from the station data
            output_path = (self.output_dir / 'raw_data' / district / station['name'])
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if file already exists
            if output_path.exists():
                self.logger.info(f"File {station['name']} already exists, skipping")
                return True

            # Download with progress tracking
            self.logger.info(f"Downloading {station['name']} from {download_url}")

            response = self.session.get(download_url, stream=True, verify=False)

            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
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
            self.metadata['download_stats']['failed_downloads'] += 1
            return False

    def run(self):
        """Main execution function"""
        if not self.login():
            self.logger.error("Login failed. Exiting...")
            return

        districts = self.get_enabled_districts()

        for district in districts:
            self.logger.info(f"Processing district {district}")

            # Get stations with retry mechanism
            stations = self.get_stations_with_retry(district)

            if not stations:
                self.logger.warning(f"No stations found for district {district} after retries")
                continue

            self.metadata['districts'][district] = {
                'total_stations': len(stations),
                'processed_stations': 0
            }
            self.metadata['download_stats']['total_stations'] += len(stations)

            # Download each file
            for station in stations:
                try:
                    success = self.download_station_data(district, station)
                    if success:
                        self.metadata['districts'][district]['processed_stations'] += 1
                    time.sleep(self.delay)
                except Exception as e:
                    self.logger.error(f"Error processing station {station['id']}: {str(e)}")

                self.save_metadata()

            self.logger.info(f"Completed district {district}")

        self.metadata["end_time"] = datetime.now().isoformat()
        self.save_metadata()
        self.logger.info("Download process complete")
        self._print_final_stats()

    def _print_final_stats(self):
        """Print final statistics"""
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
