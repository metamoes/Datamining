import re
import pandas as pd
import time
from datetime import datetime
import os
from pathlib import Path
import logging
import sys
import json
import configparser
from typing import List, Dict, Optional
import urllib3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    WebDriverException,
    NoSuchElementException
)
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PeMSDownloader:
    def __init__(self, config_path: str = "pems_config.ini"):
        """Initialize PeMS downloader using config file and Selenium"""
        # Load configuration first
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # Set credentials
        self.username = self.config['credentials']['username']
        self.password = self.config['credentials']['password']

        # Set basic parameters
        self.base_url = "https://pems.dot.ca.gov"
        self.delay = float(self.config['download_settings'].get('delay_between_requests', 2))
        self.timeout = int(self.config['download_settings'].get('timeout', 60))

        # Set output directory
        self.output_dir = Path(self.config['download_settings'].get('output_dir', 'pems_data'))

        # Setup components in correct order
        self.setup_logging()
        self.setup_directories()
        self.setup_session()
        self.setup_webdriver()
        self.init_metadata()

    def setup_session(self):
        """Configure requests session with retry mechanism"""
        self.logger.info("Setting up requests session...")

        self.session = requests.Session()

        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=10, pool_connections=10)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1'
        })

        self.logger.info("Requests session configured successfully")

    def setup_directories(self):
        """Create directory structure for outputs"""
        dirs = ['raw_data', 'processed_data', 'metadata', 'logs', 'temp_downloads']
        for dir_name in dirs:
            dir_path = self.output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def setup_logging(self):
        """Configure logging to both file and console"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_file = self.output_dir / 'logs' / 'download_log.txt'
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging initialized")

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
        self.logger.info("Metadata initialized")

    def save_metadata(self):
        """Save metadata to file"""
        try:
            metadata_path = self.output_dir / 'metadata' / 'download_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            self.logger.info("Metadata saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")

    def setup_webdriver(self):
        """Configure Selenium WebDriver with appropriate options"""
        try:
            chrome_options = Options()

            # Set Chrome options for stability and performance
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--ignore-certificate-errors')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--start-maximized')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--disable-extensions')

            # Add user agent
            chrome_options.add_argument(
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

            # Disable automation flags
            chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
            chrome_options.add_experimental_option('useAutomationExtension', False)

            # Set download directory
            downloads_path = str(self.output_dir / 'temp_downloads')
            chrome_options.add_experimental_option('prefs', {
                'download.default_directory': downloads_path,
                'download.prompt_for_download': False,
                'download.directory_upgrade': True,
                'safebrowsing.enabled': True,
                'profile.cookie_controls_mode': 0
            })

            self.logger.info("Initializing Chrome WebDriver...")

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            self.driver.implicitly_wait(20)

            self.logger.info("Chrome WebDriver initialized successfully")

        except Exception as e:
            self.logger.error(f"Error setting up WebDriver: {str(e)}")
            raise

    def login(self):
        """Login to PeMS website with improved error handling"""
        try:
            self.logger.info("Attempting to log in...")

            # Clear cookies and cache
            self.driver.delete_all_cookies()

            # Navigate to login page with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.driver.get(self.base_url)
                    self.logger.info(f"Successfully loaded login page on attempt {attempt + 1}")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load login page on attempt {attempt + 1}: {str(e)}")
                    if attempt == max_retries - 1:
                        self.logger.error(f"Failed to load login page after {max_retries} attempts")
                        return False
                    time.sleep(2)

            # Wait for page to be fully loaded
            time.sleep(5)

            # Take screenshot for debugging
            self.driver.save_screenshot(str(self.output_dir / 'logs' / 'login_page.png'))

            # Find and fill login form
            try:
                username_field = WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.NAME, 'username'))
                )
                password_field = self.driver.find_element(By.NAME, 'password')
                login_button = self.driver.find_element(By.NAME, 'login')

                username_field.clear()
                username_field.send_keys(self.username)
                time.sleep(1)

                password_field.clear()
                password_field.send_keys(self.password)
                time.sleep(1)

                login_button.click()
                time.sleep(5)

                # Take screenshot after login attempt
                self.driver.save_screenshot(str(self.output_dir / 'logs' / 'after_login.png'))

                # Verify login success
                if self.verify_login():
                    self.logger.info("Login successful")
                    # Transfer cookies to requests session
                    for cookie in self.driver.get_cookies():
                        self.session.cookies.set(cookie['name'], cookie['value'])
                    return True

                self.logger.error("Login failed - could not verify success")
                return False

            except Exception as e:
                self.logger.error(f"Error during login process: {str(e)}")
                return False

        except Exception as e:
            self.logger.error(f"Login error: {str(e)}")
            return False

    def verify_login(self) -> bool:
        """Verify successful login"""
        try:
            # Multiple checks for successful login
            success_indicators = [
                (By.XPATH, "//div[contains(text(), 'Welcome')]"),
                (By.XPATH, "//a[contains(@href, 'logout')]"),
                (By.ID, "maincontainer")  # Main container should be present after login
            ]

            for by, value in success_indicators:
                try:
                    element = self.driver.find_element(by, value)
                    if element.is_displayed():
                        self.logger.info(f"Found login success indicator: {value}")
                        return True
                except NoSuchElementException:
                    continue

            return False

        except Exception as e:
            self.logger.error(f"Error verifying login: {str(e)}")
            return False

    def get_enabled_districts(self) -> List[str]:
        """Get list of enabled districts from config"""
        districts = []
        for key, value in self.config['districts'].items():
            if value.lower() == 'true' and key.startswith('d'):
                districts.append(key[1:])  # Remove 'd' prefix
        return sorted(districts)

    def get_clearinghouse_data(self):
        """Navigate to clearinghouse and download station data"""
        try:
            # Set the target year
            target_year = datetime.now().year
            self.logger.info(f"Starting download for year {target_year}")

            # Get enabled districts
            districts = self.get_enabled_districts()

            for district in districts:
                self.logger.info(f"Processing district {district}")

                # Construct clearinghouse URL
                clearinghouse_url = (
                    f"{self.base_url}/?dnode=Clearinghouse&"
                    f"type=station_5min&district_id={district}&submit=Submit"
                )

                # Navigate to clearinghouse page
                try:
                    self.driver.get(clearinghouse_url)
                    time.sleep(3)

                    # Wait for datafiles table
                    datafiles_present = WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((By.ID, 'datafiles'))
                    )

                    if not datafiles_present:
                        self.logger.error(f"Datafiles table not found for district {district}")
                        continue

                    # Take screenshot
                    self.driver.save_screenshot(
                        str(self.output_dir / 'logs' / f'clearinghouse_d{district}.png')
                    )

                    # Find all download links
                    download_links = self.driver.find_elements(
                        By.CSS_SELECTOR,
                        '#datafiles a[href*="download"]'
                    )

                    self.logger.info(f"Found {len(download_links)} download links")

                    # Process each download link
                    for link in download_links:
                        try:
                            href = link.get_attribute('href')
                            filename = link.text.strip()

                            self.logger.debug(f"Processing link: {href}")
                            self.logger.debug(f"Filename: {filename}")

                            if self._is_valid_file(filename, district, target_year):
                                self.logger.info(f"Processing file: {filename}")

                                download_id = self._extract_download_id(href)
                                if not download_id:
                                    continue

                                success = self._download_file(
                                    download_id, filename, district
                                )

                                if success:
                                    self.metadata['download_stats']['successful_downloads'] += 1
                                else:
                                    self.metadata['download_stats']['failed_downloads'] += 1

                                self.save_metadata()
                                time.sleep(self.delay)
                            else:
                                self.logger.debug(f"Skipping file {filename} - does not match pattern")

                        except Exception as e:
                            self.logger.error(f"Error processing link: {str(e)}")
                            continue

                except Exception as e:
                    self.logger.error(f"Error accessing clearinghouse for district {district}: {str(e)}")
                    continue

        except Exception as e:
            self.logger.error(f"Error in get_clearinghouse_data: {str(e)}")

    def _is_valid_file(self, filename: str, district: str, year: int) -> bool:
        """
        Check if file matches target pattern and year

        Args:
            filename: Name of the file to check
            district: District number as string
            year: Target year

        Returns:
            bool: True if file matches pattern, False otherwise
        """
        try:
            # Ensure district is padded with leading zero if needed
            district_padded = district.zfill(2)
            pattern = rf'd{district_padded}_text_station_5min_{year}_\d{{2}}_\d{{2}}\.txt\.gz'

            self.logger.debug(f"Checking file {filename} against pattern {pattern}")
            match = bool(re.match(pattern, filename))

            if match:
                self.logger.debug(f"File {filename} matches pattern")
            else:
                self.logger.debug(f"File {filename} does not match pattern")

            return match

        except Exception as e:
            self.logger.error(f"Error checking file pattern: {str(e)}")
            return False

    def _extract_download_id(self, href: str) -> Optional[str]:
        """Extract download ID from href"""
        try:
            match = re.search(r'download=(\d+)', href)
            return match.group(1) if match else None
        except Exception as e:
            self.logger.error(f"Error extracting download ID: {str(e)}")
            return None

    def _download_file(self, download_id: str, filename: str, district: str) -> bool:
        """Download individual file from clearinghouse"""
        try:
            # Construct download URL
            download_url = f"{self.base_url}/?download={download_id}&dnode=Clearinghouse"

            # Set up output path
            output_path = self.output_dir / 'raw_data' / f'district_{district}' / filename

            # Skip if file already exists
            if output_path.exists():
                self.logger.info(f"File already exists: {filename}")
                return True

            # Create directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"Downloading {filename} (Attempt {attempt + 1}/{max_retries})")

                    response = self.session.get(
                        download_url,
                        stream=True,
                        verify=False,
                        timeout=self.timeout
                    )

                    if response.status_code == 200:
                        total_size = int(response.headers.get('content-length', 0))
                        block_size = 8192
                        downloaded_size = 0

                        with open(output_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=block_size):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)

                                    # Log progress for large files
                                    if total_size > 0:
                                        progress = (downloaded_size / total_size) * 100
                                        if downloaded_size % (5 * block_size) == 0:
                                            self.logger.info(
                                                f"Download progress for {filename}: "
                                                f"{progress:.1f}%"
                                            )

                        self.logger.info(f"Successfully downloaded {filename}")
                        return True
                    else:
                        self.logger.warning(
                            f"Download failed with status {response.status_code} "
                            f"(Attempt {attempt + 1}/{max_retries})"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Error downloading {filename} "
                        f"(Attempt {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(self.delay)
                        continue

            return False

        except Exception as e:
            self.logger.error(f"Error in _download_file: {str(e)}")
            return False

    def run(self):
        """Main execution function"""
        try:
            self.logger.info("Starting PeMS data collection process...")

            # Attempt login
            if not self.login():
                self.logger.error("Login failed. Exiting...")
                return False

            self.logger.info("Successfully logged in. Starting data collection...")

            # Get data from clearinghouse
            self.get_clearinghouse_data()

            # Record completion time
            self.metadata["end_time"] = datetime.now().isoformat()
            self.save_metadata()

            self.logger.info("Download process complete")
            self._print_final_stats()

            return True

        except Exception as e:
            self.logger.error(f"Error in run: {str(e)}")
            return False
        finally:
            self.cleanup()

    def _print_final_stats(self):
        """Print final statistics"""
        total_downloads = (
                self.metadata['download_stats']['successful_downloads'] +
                self.metadata['download_stats']['failed_downloads']
        )

        success_rate = (
            self.metadata['download_stats']['successful_downloads'] /
            total_downloads * 100 if total_downloads > 0 else 0
        )

        self.logger.info("\nFinal Statistics:")
        self.logger.info(f"Total files processed: {total_downloads}")
        self.logger.info(
            f"Successful downloads: "
            f"{self.metadata['download_stats']['successful_downloads']}"
        )
        self.logger.info(
            f"Failed downloads: "
            f"{self.metadata['download_stats']['failed_downloads']}"
        )
        self.logger.info(f"Success rate: {success_rate:.2f}%")

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
                self.logger.info("WebDriver cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def main():
    """Main entry point for the script"""
    try:
        # Create and run downloader
        with PeMSDownloader("pems_config.ini") as downloader:
            success = downloader.run()

            if success:
                print("\nData collection completed successfully!")
                print("Check the logs directory for detailed information.")
            else:
                print("\nData collection completed with errors.")
                print("Please check the log files for details.")

    except Exception as e:
        print(f"\nError during script execution: {str(e)}")
        print("\nPlease check:")
        print("1. Your internet connection is stable")
        print("2. The PeMS website is accessible")
        print("3. Your credentials in pems_config.ini are correct")
        print("4. Chrome browser is properly installed")
        print("5. Check the logs directory for detailed error information")


if __name__ == "__main__":
    main()
