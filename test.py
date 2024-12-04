import os
import shutil

def delete_pycache_in_dir(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        # Check for '__pycache__' directories
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                print(f"Deleting {pycache_path}")
                shutil.rmtree(pycache_path)  # Delete the __pycache__ directory and its contents
            except Exception as e:
                print(f"Failed to delete {pycache_path}: {e}")

if __name__ == "__main__":
    current_directory = os.getcwd()  # Get the current working directory
    delete_pycache_in_dir(current_directory)  # Delete all __pycache__ directories
