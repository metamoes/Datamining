import os
from pathlib import Path
import re
import logging
from datetime import datetime
import pandas as pd

class PeMSDataCombiner:
    def __init__(self, base_dir: str = "pems_data"):
        """
        Initialize the data combiner
        Args:
            base_dir: Base directory where PeMS data is stored
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw_data'
        self.processed_dir = self.base_dir / 'processed_data'
        self.setup_logging()

    def setup_logging(self):
        """Configure logging"""
        log_dir = self.base_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'combine_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def extract_date_from_filename(self, filename: str) -> datetime:
        """
        Extract date from filename in format d##_text_station_5min_YYYY_MM_DD.txt
        """
        try:
            match = re.search(r'(\d{4})_(\d{2})_(\d{2})\.txt$', filename)
            if match:
                year, month, day = map(int, match.groups())
                return datetime(year, month, day)
        except Exception as e:
            self.logger.error(f"Error parsing date from filename {filename}: {e}")
        return None

    def process_header(self, first_file: Path) -> str:
        """
        Read and validate header from first file
        """
        try:
            with open(first_file, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
            return header
        except Exception as e:
            self.logger.error(f"Error reading header from {first_file}: {e}")
            return None

    def combine_district_files(self, district_dir: Path, output_file: Path):
        """
        Combine all files for a single district in chronological order
        """
        try:
            # Get all .txt files in district directory
            files = [f for f in district_dir.glob('*.txt')]
            
            if not files:
                self.logger.warning(f"No text files found in {district_dir}")
                return
            
            # Sort files by date
            sorted_files = sorted(
                files,
                key=lambda x: self.extract_date_from_filename(x.name)
            )
            
            # Get header from first file
            header = self.process_header(sorted_files[0])
            if not header:
                return
            
            # Create output directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write header to output file
            with open(output_file, 'w', encoding='utf-8') as outfile:
                outfile.write(header + '\n')
                
                # Process each file
                total_rows = 0
                for idx, file in enumerate(sorted_files, 1):
                    try:
                        self.logger.info(f"Processing file {idx}/{len(sorted_files)}: {file.name}")
                        
                        # Read file, skip header, and append to output
                        with open(file, 'r', encoding='utf-8') as infile:
                            next(infile)  # Skip header
                            file_rows = 0
                            for line in infile:
                                outfile.write(line)
                                file_rows += 1
                        
                        total_rows += file_rows
                        self.logger.info(f"Added {file_rows:,} rows from {file.name}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing file {file}: {e}")
            
            self.logger.info(f"Completed {output_file.name} with {total_rows:,} total rows")
            return total_rows
            
        except Exception as e:
            self.logger.error(f"Error combining files for {district_dir}: {e}")
            return 0

    def combine_all_districts(self):
        """
        Process all district directories and combine their files
        """
        try:
            # Get all district directories
            district_dirs = [d for d in self.raw_dir.glob('district_*') if d.is_dir()]
            
            if not district_dirs:
                self.logger.error(f"No district directories found in {self.raw_dir}")
                return
            
            # Process each district
            total_stats = {
                'districts_processed': 0,
                'total_rows': 0,
                'start_time': datetime.now()
            }
            
            for district_dir in sorted(district_dirs):
                district_num = district_dir.name.split('_')[1]
                output_file = self.processed_dir / f'district_{district_num}_combined.txt'
                
                self.logger.info(f"\nProcessing {district_dir.name}")
                rows = self.combine_district_files(district_dir, output_file)
                
                if rows > 0:
                    total_stats['districts_processed'] += 1
                    total_stats['total_rows'] += rows
            
            # Log summary statistics
            total_stats['end_time'] = datetime.now()
            duration = total_stats['end_time'] - total_stats['start_time']
            
            self.logger.info("\nProcessing Complete!")
            self.logger.info(f"Districts Processed: {total_stats['districts_processed']}")
            self.logger.info(f"Total Rows: {total_stats['total_rows']:,}")
            self.logger.info(f"Duration: {duration}")
            
            # Save summary to file
            summary_file = self.processed_dir / 'processing_summary.txt'
            with open(summary_file, 'w') as f:
                f.write("PeMS Data Processing Summary\n")
                f.write("==========================\n\n")
                f.write(f"Start Time: {total_stats['start_time']}\n")
                f.write(f"End Time: {total_stats['end_time']}\n")
                f.write(f"Duration: {duration}\n")
                f.write(f"Districts Processed: {total_stats['districts_processed']}\n")
                f.write(f"Total Rows: {total_stats['total_rows']:,}\n")
            
        except Exception as e:
            self.logger.error(f"Error in combine_all_districts: {e}")

def main():
    """Main execution function"""
    try:
        print("PeMS Data Combiner")
        print("=================")
        
        # Get base directory
        default_dir = "pems_data"
        base_dir = input(f"Enter base directory path (press Enter for '{default_dir}'): ").strip()
        if not base_dir:
            base_dir = default_dir
        
        # Create and run combiner
        combiner = PeMSDataCombiner(base_dir)
        combiner.combine_all_districts()
        
        print("\nProcessing complete! Check the logs directory for detailed information.")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        print("Please check the logs for detailed error information.")

if __name__ == "__main__":
    main()
