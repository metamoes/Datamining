import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrafficDataPreprocessor:
    def __init__(self, input_dir: Union[str, Path], output_dir: Union[str, Path]):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Required normalized features for the model
        self.required_features = [
            'Is_Peak_Hour_Normalized',
            'Station_Length_Normalized',
            'Total_Flow_Normalized',
            'Avg_Occupancy_Normalized',
            'Avg_Speed_Normalized',
            'Direction_S_Normalized',
            'Direction_E_Normalized',
            'Direction_W_Normalized',
            'Lane_Type_FR_Normalized',
            'Lane_Type_ML_Normalized',
            'Lane_Type_OR_Normalized',
            'Active_Lanes_Normalized',
            'Lane_1_Flow_Normalized',
            'Lane_1_Avg_Occ_Normalized',
            'Lane_1_Avg_Speed_Normalized',
            'Lane_1_Efficiency_Normalized',
            'Lane_2_Flow_Normalized',
            'Lane_2_Avg_Occ_Normalized',
            'Lane_2_Avg_Speed_Normalized',
            'Lane_2_Efficiency_Normalized',
            'Lane_3_Flow_Normalized',
            'Lane_3_Avg_Occ_Normalized',
            'Lane_3_Avg_Speed_Normalized',
            'Lane_3_Efficiency_Normalized',
            'Lane_4_Flow_Normalized',
            'Lane_4_Avg_Occ_Normalized',
            'Lane_4_Avg_Speed_Normalized',
            'Lane_4_Efficiency_Normalized'
        ]

        # Temporal features
        self.temporal_features = [
            'Hour',
            'Day_of_Week',
            'Is_Weekend',
            'Month'
        ]

        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_rows': 0,
            'rows_with_missing': 0
        }

    def detect_file_format(self, file_path: Path) -> Dict[str, str]:
        """
        Detect the format of the input file by trying different read methods.
        Returns a dictionary with the successful reading parameters.
        """
        try:
            # Try reading first few lines to determine format
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()

            # Try different delimiters
            delimiters = ['\t', ',', ';', '|']
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(file_path, sep=delimiter, nrows=5)
                    if len(df.columns) > 1:  # Successfully parsed with multiple columns
                        logger.info(f"Detected delimiter: '{delimiter}' for {file_path.name}")
                        return {'separator': delimiter, 'encoding': 'utf-8'}
                except:
                    continue

            raise ValueError(f"Could not determine file format for {file_path.name}")

        except Exception as e:
            logger.error(f"Error detecting file format: {str(e)}")
            return None

    def read_file(self, file_path: Path) -> pd.DataFrame:
        """
        Read file with automatic format detection.
        """
        try:
            # First detect the file format
            format_params = self.detect_file_format(file_path)
            if not format_params:
                return None

            # Read the file with detected parameters
            df = pd.read_csv(
                file_path,
                sep=format_params['separator'],
                encoding=format_params['encoding']
            )

            return df

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

    def validate_features(self, df: pd.DataFrame) -> bool:
        """Validate that all required features are present."""
        missing_features = set(self.required_features) - set(df.columns)
        if missing_features:
            logger.error(f"Missing required normalized features: {missing_features}")
            return False

        missing_temporal = set(self.temporal_features) - set(df.columns)
        if missing_temporal:
            logger.error(f"Missing temporal features: {missing_temporal}")
            return False

        return True

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Count missing values before processing
        missing_before = df.isnull().sum()
        if missing_before.any():
            self.stats['rows_with_missing'] += missing_before.sum()
            logger.info(f"Found columns with missing values: {missing_before[missing_before > 0]}")

        # Handle missing values for each type of feature
        for col in df.columns:
            if df[col].isnull().any():
                if 'Normalized' in col:
                    if 'Flow' in col:
                        df[col] = df[col].fillna(0)
                    elif any(x in col for x in ['Speed', 'Occupancy', 'Efficiency']):
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(0)
                elif col in self.temporal_features:
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(0)

        return df

    def process_file(self, file_path: Path) -> bool:
        """Process a single file."""
        try:
            # Read file with format detection
            df = self.read_file(file_path)
            if df is None:
                return False

            original_rows = len(df)
            self.stats['total_rows'] += original_rows

            # Validate features
            if not self.validate_features(df):
                logger.error(f"Feature validation failed for {file_path.name}")
                return False

            # Handle missing values
            df = self.handle_missing_values(df)

            # Verify no missing values remain in required features
            missing_after = df[self.required_features + self.temporal_features].isnull().sum()
            if missing_after.any():
                logger.error(
                    f"Remaining missing values after processing in {file_path.name}: {missing_after[missing_after > 0]}")
                return False

            # Create output directory structure
            relative_path = file_path.relative_to(self.input_dir)
            output_path = self.output_dir / relative_path.parent / f"validated_{relative_path.name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save processed file
            df.to_csv(output_path, index=False)
            self.stats['processed_files'] += 1
            logger.info(f"Successfully processed {file_path.name}: {original_rows} rows")
            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            self.stats['failed_files'] += 1
            return False

    def analyze_sample_file(self, file_path: Path):
        """Analyze and print detailed information about a sample file."""
        try:
            # Try different reading methods
            logger.info(f"\nAnalyzing file: {file_path.name}")

            # Read raw content
            with open(file_path, 'r', encoding='utf-8') as f:
                first_few_lines = [next(f) for _ in range(5)]

            logger.info("First few lines of raw content:")
            for line in first_few_lines:
                logger.info(line.strip())

            # Try reading with different delimiters
            delimiters = ['\t', ',', ';', '|']
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(file_path, sep=delimiter, nrows=5)
                    logger.info(f"\nTrying delimiter: '{delimiter}'")
                    logger.info(f"Columns found: {len(df.columns)}")
                    logger.info(f"Column names: {df.columns.tolist()}")
                except Exception as e:
                    logger.info(f"Failed with delimiter '{delimiter}': {str(e)}")

        except Exception as e:
            logger.error(f"Error analyzing file: {str(e)}")

    def process_all_files(self):
        """Process all files in the input directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find all input files
        input_files = list(self.input_dir.rglob("*.csv"))
        self.stats['total_files'] = len(input_files)

        logger.info(f"Found {len(input_files)} files to process")

        # Analyze first file to understand structure
        if input_files:
            self.analyze_sample_file(input_files[0])

        # Process files with progress bar
        with tqdm(total=len(input_files), desc="Processing files") as pbar:
            for file_path in input_files:
                success = self.process_file(file_path)
                status = "Processed" if success else "Failed"
                pbar.set_description(f"{status}: {file_path.name}")
                pbar.update(1)

        # Log final statistics
        logger.info("\nProcessing Summary:")
        logger.info(f"Total files found: {self.stats['total_files']}")
        logger.info(f"Successfully processed: {self.stats['processed_files']}")
        logger.info(f"Failed: {self.stats['failed_files']}")
        logger.info(f"Total rows processed: {self.stats['total_rows']}")
        logger.info(f"Rows with missing values handled: {self.stats['rows_with_missing']}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Validate and preprocess traffic data files.')
    parser.add_argument('input_dir', type=str, help='Input directory containing CSV files')
    parser.add_argument('output_dir', type=str, help='Output directory for processed files')

    args = parser.parse_args()

    try:
        preprocessor = TrafficDataPreprocessor(args.input_dir, args.output_dir)
        preprocessor.process_all_files()

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()