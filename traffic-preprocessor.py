import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from datetime import datetime
from pathlib import Path


class TrafficDataPreprocessor:
    def __init__(self):
        # Initialize encoders with updated parameters
        self.direction_encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.lane_type_encoder = OneHotEncoder(sparse_output=False, drop='first')

        # Pre-fit direction encoder with all possible values
        self.direction_encoder.fit(np.array(['N', 'S', 'E', 'W']).reshape(-1, 1))
        self.lane_type_encoder.fit(np.array(['CD', 'CH', 'FF', 'FR', 'HV', 'ML', 'OR']).reshape(-1, 1))

        # Define column names
        self.columns = [
            'Timestamp', 'Station', 'District', 'Freeway', 'Direction', 'Lane_Type',
            'Station_Length', 'Samples', 'Pct_Observed', 'Total_Flow', 'Avg_Occupancy',
            'Avg_Speed'
        ]

        # Add lane-specific columns
        for lane in range(1, 9):
            self.columns.extend([
                f'Lane_{lane}_Samples',
                f'Lane_{lane}_Flow',
                f'Lane_{lane}_Avg_Occ',
                f'Lane_{lane}_Avg_Speed',
                f'Lane_{lane}_Observed'
            ])

        # Define timestamp format
        self.timestamp_format = '%m/%d/%Y %H:%M:%S'

    def _read_file(self, file_path):
        """Read the file with proper handling of delimiters."""
        try:
            # First try reading as CSV with comma delimiter
            df = pd.read_csv(file_path, header=None)

            # If we got just one column, the file might be tab-delimited
            if len(df.columns) == 1:
                # Try reading with tab delimiter
                df = pd.read_csv(file_path, header=None, delimiter='\t')

            # Assign column names
            df.columns = self.columns[:len(df.columns)]

            return df

        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None

    def _preprocess_file(self, file_path):
        """Process a single file."""
        try:
            # Read the file
            df = self._read_file(file_path)
            if df is None:
                return None

            print(f"Successfully read file with shape: {df.shape}")

            # Initialize dictionary to store all transformed data
            transformed_data = {}

            # Convert timestamp with specific format
            transformed_data['Timestamp'] = pd.to_datetime(df['Timestamp'], format=self.timestamp_format)
            transformed_data['Hour'] = transformed_data['Timestamp'].dt.hour
            transformed_data['Day_of_Week'] = transformed_data['Timestamp'].dt.dayofweek
            transformed_data['Is_Weekend'] = transformed_data['Timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
            transformed_data['Month'] = transformed_data['Timestamp'].dt.month
            transformed_data['Year'] = transformed_data['Timestamp'].dt.year
            transformed_data['Is_Peak_Hour'] = transformed_data['Timestamp'].dt.hour.isin([7, 8, 9, 16, 17, 18]).astype(
                int)

            # Process station information
            transformed_data['Station'] = df['Station']
            transformed_data['District'] = df['District']
            transformed_data['Freeway'] = df['Freeway']

            # Convert numeric columns and handle missing values
            numeric_columns = [
                'Station_Length', 'Samples', 'Pct_Observed', 'Total_Flow',
                'Avg_Occupancy', 'Avg_Speed'
            ]

            # Add lane-specific numeric columns
            lane_cols = [col for col in df.columns if col.startswith('Lane_')]
            numeric_columns.extend(lane_cols)

            # Process numeric columns
            for col in numeric_columns:
                if col in df.columns:
                    transformed_data[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

            # Create DataFrame with transformed data so far
            result_df = pd.DataFrame(transformed_data)

            # Process categorical variables
            dir_encoded = self.direction_encoder.transform(df['Direction'].fillna('N').values.reshape(-1, 1))
            lane_encoded = self.lane_type_encoder.transform(df['Lane_Type'].fillna('ML').values.reshape(-1, 1))

            # Add encoded categorical variables
            for i, col in enumerate(['S', 'E', 'W']):
                result_df[f'Direction_{col}'] = dir_encoded[:, i]

            for i, col in enumerate(['CH', 'FF', 'FR', 'HV', 'ML', 'OR']):
                result_df[f'Lane_Type_{col}'] = lane_encoded[:, i]

            # Calculate lane efficiencies
            lane_flow_cols = [col for col in result_df.columns if 'Lane_' in col and 'Flow' in col]
            efficiency_data = {}

            for flow_col in lane_flow_cols:
                speed_col = flow_col.replace('Flow', 'Avg_Speed')
                efficiency_col = flow_col.replace('Flow', 'Efficiency')
                if speed_col in result_df.columns:
                    efficiency_data[efficiency_col] = result_df[flow_col] * result_df[speed_col]

            # Add efficiency data
            for col, data in efficiency_data.items():
                result_df[col] = data

            # Calculate aggregate metrics
            result_df['Total_Lane_Efficiency'] = sum(efficiency_data.values())
            result_df['Active_Lanes'] = (result_df[lane_flow_cols] > 0).sum(axis=1)

            # Normalize numeric features
            numeric_features = result_df.select_dtypes(include=[np.number]).columns
            normalized_data = {}

            for col in numeric_features:
                if col not in ['Station', 'District', 'Freeway', 'Year', 'Month', 'Day_of_Week', 'Hour']:
                    std = result_df[col].std()
                    if std > 0:
                        normalized_data[f'{col}_Normalized'] = (result_df[col] - result_df[col].mean()) / std

            # Add normalized columns
            result_df = pd.concat([result_df, pd.DataFrame(normalized_data)], axis=1)

            return result_df

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            print("Error details:", str(e.__class__.__name__))
            return None

    def process_directory(self, input_dir, output_dir):
        """Process all files in a directory and its subdirectories."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_count = 0
        error_count = 0
        skipped_count = 0

        # Get all subdirectories
        all_dirs = [x for x in input_dir.rglob('*') if x.is_dir()]

        # Process each directory
        for current_dir in [input_dir] + all_dirs:
            print(f"\nProcessing directory: {current_dir}")

            # Create corresponding output directory
            relative_path = current_dir.relative_to(input_dir)
            current_output_dir = output_dir / relative_path
            current_output_dir.mkdir(parents=True, exist_ok=True)

            # Process each CSV file
            for file_path in current_dir.glob('*.csv'):
                try:
                    output_file_path = current_output_dir / f"processed_{file_path.name}"

                    if output_file_path.exists():
                        print(f"Skipping already processed file: {file_path}")
                        skipped_count += 1
                        continue

                    print(f"\nProcessing file: {file_path}")
                    df = self._preprocess_file(file_path)

                    if df is not None:
                        df.to_csv(output_file_path, index=False)
                        processed_count += 1
                        print(f"Successfully processed: {file_path}")
                    else:
                        error_count += 1

                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    error_count += 1

        return processed_count, error_count, skipped_count


def main():
    input_directory = input("Enter the path to the input directory: ").strip()
    output_directory = input("Enter the path to the output directory (or press Enter to use input directory): ").strip()

    if not output_directory:
        output_directory = input_directory

    preprocessor = TrafficDataPreprocessor()

    print("\nStarting preprocessing...")
    processed, errors, skipped = preprocessor.process_directory(input_directory, output_directory)

    print("\nPreprocessing Summary:")
    print(f"Processed files: {processed}")
    print(f"Skipped files: {skipped}")
    print(f"Errors encountered: {errors}")
    print(f"Output directory: {output_directory}")


if __name__ == "__main__":
    main()