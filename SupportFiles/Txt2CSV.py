import os
import glob


def convert_txt_to_csv(txt_path, csv_path=None):

    if csv_path is None:
        # Replace .txt extension with .csv
        csv_path = txt_path.rsplit('.', 1)[0] + '.csv'

    try:
        # Read txt file and write to csv
        with open(txt_path, 'r') as txt_file, open(csv_path, 'w') as csv_file:
            for line in txt_file:
                csv_file.write(line)

        print(f"Successfully converted: {txt_path} → {csv_path}")
    except Exception as e:
        print(f"Error converting {txt_path}: {str(e)}")


def find_and_convert_txt_files(root_dir):
    """
    Recursively find and convert all .txt files in a directory and its subdirectories

    Args:
        root_dir (str): Root directory to start the search
    """
    # Ensure root_dir is absolute path
    root_dir = os.path.abspath(root_dir)

    # Counter for converted files
    converted_count = 0
    error_count = 0

    print(f"Searching for .txt files in {root_dir} and its subdirectories...")

    # Walk through directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Find all .txt files in current directory
        for filename in filenames:
            if filename.endswith('.txt'):
                txt_path = os.path.join(dirpath, filename)
                try:
                    convert_txt_to_csv(txt_path)
                    converted_count += 1
                except Exception as e:
                    print(f"Error processing {txt_path}: {str(e)}")
                    error_count += 1

    # Print summary
    print("\nConversion Summary:")
    print(f"Total .txt files found: {converted_count + error_count}")
    print(f"Successfully converted: {converted_count}")
    print(f"Errors encountered: {error_count}")


if __name__ == "__main__":
    # Replace with your directory path
    directory_path = "./data"

    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist")
    else:
        find_and_convert_txt_files(directory_path)