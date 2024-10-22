import os
import configparser

def check_config():
    # Get the current working directory
    current_dir = os.getcwd()
    config_path = os.path.join(current_dir, "pems_config.ini")
    
    print(f"Current working directory: {current_dir}")
    print(f"Looking for config file at: {config_path}")
    
    if os.path.exists(config_path):
        print("Config file found!")
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            print("\nConfig sections found:")
            for section in config.sections():
                print(f"- {section}")
        except Exception as e:
            print(f"Error reading config: {str(e)}")
    else:
        print("Config file not found!")
        print("\nDirectory contents:")
        for file in os.listdir(current_dir):
            print(f"- {file}")

if __name__ == "__main__":
    check_config()
