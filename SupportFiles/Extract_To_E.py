import os
import gzip
import shutil
from pathlib import Path
import hashlib


def get_file_hash(filepath):
    """Calculate MD5 hash of a file for comparison."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_extracted_content_hash(gz_filepath):
    """Calculate hash of the content inside the gz file."""
    hash_md5 = hashlib.md5()
    with gzip.open(gz_filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def extract_gz_files(start_path):
    """
    Walk through all subdirectories starting from start_path,
    find .gz files and extract them to E: drive maintaining the same structure.
    Verifies if files are already extracted and removes source .txt after successful extraction.

    Args:
        start_path (str): The root directory to start searching from
    """
    # Convert the start path to absolute path and get the base folder name
    start_path = os.path.abspath(start_path)
    base_path = os.path.basename(start_path)

    # Counters for processed files
    processed = 0
    skipped = 0
    errors = 0
    deleted = 0

    print(f"Starting search in: {start_path}")
    print(f"Files will be extracted to E: drive")

    # Walk through all subdirectories
    for root, dirs, files in os.walk(start_path):
        # Filter for .gz files
        gz_files = [f for f in files if f.endswith('.gz')]

        for gz_file in gz_files:
            gz_path = os.path.join(root, gz_file)

            # Calculate relative path from start_path
            rel_path = os.path.relpath(root, start_path)

            # Create the new path on E: drive
            new_root = os.path.join('E:\\', base_path, rel_path)

            # Get the output filename by removing .gz extension
            output_file = os.path.join(new_root, gz_file[:-3])

            try:
                print(f"\nProcessing: {gz_path}")

                # Check if file already exists in destination
                if os.path.exists(output_file):
                    # Compare content hashes to verify if it's already correctly extracted
                    try:
                        existing_hash = get_file_hash(output_file)
                        gz_content_hash = get_extracted_content_hash(gz_path)

                        if existing_hash == gz_content_hash:
                            print(f"File already exists and content matches at: {output_file}")
                            skipped += 1

                            # Check if there's a .txt file to delete
                            txt_file = gz_path[:-3]
                            if os.path.exists(txt_file):
                                os.remove(txt_file)
                                print(f"Deleted existing .txt file: {txt_file}")
                                deleted += 1

                            continue
                        else:
                            print("Existing file has different content - will extract and overwrite")
                    except Exception as e:
                        print(f"Error comparing files: {str(e)} - will re-extract")

                # Create the directory structure if it doesn't exist
                os.makedirs(new_root, exist_ok=True)

                print(f"Extracting to: {output_file}")

                # Extract the file
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Check if extraction was successful by comparing hashes
                if get_file_hash(output_file) == get_extracted_content_hash(gz_path):
                    processed += 1
                    print("Extraction successful!")

                    # Delete the .txt file if it exists
                    txt_file = gz_path[:-3]
                    if os.path.exists(txt_file):
                        os.remove(txt_file)
                        print(f"Deleted source .txt file: {txt_file}")
                        deleted += 1
                else:
                    raise Exception("Extracted file content does not match source")

            except Exception as e:
                errors += 1
                print(f"Error processing {gz_path}: {str(e)}")

    # Print summary
    print("\nExtraction Complete!")
    print(f"Files processed successfully: {processed}")
    print(f"Files skipped (already extracted): {skipped}")
    print(f"Source .txt files deleted: {deleted}")
    if errors > 0:
        print(f"Files with errors: {errors}")


if __name__ == "__main__":
    # Get the current directory as the default start path
    current_dir = os.getcwd()

    print("GZ File Extractor (to E: drive)")
    print("-------------------------------")

    # Ask user if they want to use a different directory
    user_input = input(f"Press Enter to use current directory ({current_dir})\nor enter a different path: ").strip()

    start_path = user_input if user_input else current_dir

    # Verify the path exists
    if os.path.exists(start_path):
        # Verify E: drive exists
        if os.path.exists('E:\\'):
            extract_gz_files(start_path)
        else:
            print("Error: E: drive not found!")
    else:
        print(f"Error: Path '{start_path}' does not exist!")
