import os
import sys
import shutil

# Ensure we can import the tool
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import download_ood_datasets

# Mock the URL dict to use a small file for testing
# Using a small repo zip from verified GitHub location (e.g. the repo itself or a small one)
# Here we use the repo's own archive for testing if possible, or a known small file.
# Since we need a "zip" that extracts to a folder, let's use a dummy zip creation or a small external one.
# For stability, let's just use the 'requests' library zip as a test or similar. 
# Actually, the user wants "local simple test". 
# Let's mock the request response to avoid actual network if possible? 
# "local simple test" usually means run the code. "no need to download completely" implies we can stop early or use small file.
# Using a small file is better to verify extraction logic.

TEST_URL = "https://github.com/octocat/Spoon-Knife/archive/refs/heads/main.zip"

# Monkey patch the dictionary in the module
download_ood_datasets.DOWNLOAD_URL_DICT = {
    'inaturalist': TEST_URL,
    'openimage_o': TEST_URL
}

# Temporary test directory
TEST_DIR = "data_test_local"

def run_test():
    print(f"Starting local simple test using {TEST_URL}...")
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    
    # Run the download function
    try:
        # Test inaturalist download (mapped to small zip)
        # It should download into data_test_local/images_largescale/inaturalist
        print("Testing download_and_extract for 'inaturalist'...")
        download_ood_datasets.download_and_extract('inaturalist', TEST_DIR)
        
        # Verify
        rel_path = download_ood_datasets.DIR_DICT['inaturalist']
        expected_path = os.path.join(TEST_DIR, rel_path)
        
        # The zip from github (Spoon-Knife-main) extracts to a folder "Spoon-Knife-main".
        # The script extracts into `dest_dir`. 
        # So we expect `data_test_local/images_largescale/inaturalist/Spoon-Knife-main`...
        
        if os.path.exists(expected_path):
            print("\n[SUCCESS] Destination directory created.")
            print(f"Contents of {expected_path}:")
            items = os.listdir(expected_path)
            print(items)
            if items:
                print("[SUCCESS] Files extracted.")
            else:
                print("[WARNING] Directory empty - extraction might have failed or zip structure mismatch.")
        else:
            print(f"\n[FAILURE] Expected directory {expected_path} not found.")

    except Exception as e:
        print(f"\n[FAILURE] Test failed with exception: {e}")
    finally:
        # Cleanup
        if os.path.exists(TEST_DIR):
            print(f"Cleaning up {TEST_DIR}...")
            shutil.rmtree(TEST_DIR)

if __name__ == "__main__":
    run_test()
