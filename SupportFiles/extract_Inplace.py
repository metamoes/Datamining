import os
import gzip
import shutil
from pathlib import Path

def extract_gz_files(start_path):
    """
    Walk through all subdirectories starting from start_path,
    find .gz files and extract them to the same location.
    
    Args:
        start_path (str): The root directory to start searching from
    """
    # Convert the start path to absolute path
    start_path = os.path.abspath(start_path)
    
    # Counter for processed files
    processed = 0
    errors = 0
    
    print(f"Starting search in: {start_path}")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(start_path):
        # Filter for .gz files
        gz_files = [f for f in files if f.endswith('.gz')]
        
        for gz_file in gz_files:
            gz_path = os.path.join(root, gz_file)
            # Get the output filename by removing .gz extension
            output_file = gz_path[:-3]
            
            try:
                print(f"Extracting: {gz_path}")
                
                # Open the .gz file and extract it
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                processed += 1
                print(f"Successfully extracted to: {output_file}")
                
            except Exception as e:
                errors += 1
                print(f"Error processing {gz_path}: {str(e)}")
    
    # Print summary
    print("\nExtraction Complete!")
    print(f"Files processed successfully: {processed}")
    if errors > 0:
        print(f"Files with errors: {errors}")

if __name__ == "__main__":
    # Get the current directory as the default start path
    current_dir = os.getcwd()
    
    print("GZ File Extractor")
    print("----------------")
    
    # Ask user if they want to use a different directory
    user_input = input(f"Press Enter to use current directory ({current_dir})\nor enter a different path: ").strip()
    
    start_path = user_input if user_input else current_dir
    
    # Verify the path exists
    if os.path.exists(start_path):
        extract_gz_files(start_path)
    else:
        print(f"Error: Path '{start_path}' does not exist!")
