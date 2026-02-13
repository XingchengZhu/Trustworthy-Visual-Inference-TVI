import os
import argparse
# import gdown (moved to local scope)
import zipfile

# OpenOOD Benchmark IDs
# GitHub Release URLs (Placeholders - Please update with actual URLs)
DOWNLOAD_URL_DICT = {
    'inaturalist': 'https://github.com/Jingkang50/OpenOOD/releases/download/v1.5/inaturalist.zip',  # TODO: Update this
    'openimage_o': 'https://github.com/Jingkang50/OpenOOD/releases/download/v1.5/openimage_o.zip',  # TODO: Update this
}

# Target Directories relative to data/
DIR_DICT = {
    'places365': 'images_classic/places365',
    'texture': 'images_classic/texture',
    'inaturalist': 'images_largescale/inaturalist',
    'openimage_o': 'images_largescale/openimage_o',
}

def download_file_from_url(url, dest_path):
    import requests
    from tqdm import tqdm
    
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
        return False
    return True

def download_and_extract(dataset_name, data_root, source_dir=None):
    if dataset_name not in DIR_DICT:
        print(f"Dataset {dataset_name} not supported.")
        return

    rel_path = DIR_DICT[dataset_name]
    dest_dir = os.path.join(data_root, rel_path)
    if os.path.exists(dest_dir) and os.listdir(dest_dir):
        print(f"Dataset {dataset_name} already exists at {dest_dir}. Skipping.")
        return

    os.makedirs(dest_dir, exist_ok=True)
    
    # Check for local source first
    local_zip_path = None
    if source_dir:
        potential_zip = os.path.join(source_dir, f"{dataset_name}.zip")
        if os.path.exists(potential_zip):
            print(f"Found local dataset source for {dataset_name} at {potential_zip}")
            local_zip_path = potential_zip
        else:
            print(f"Local source specified but {potential_zip} not found. Attempting download...")

    if local_zip_path:
        zip_path = local_zip_path
        should_remove_zip = False 
    else:
        if dataset_name not in DOWNLOAD_URL_DICT:
             print(f"No download URL available for {dataset_name}")
             return

        download_url = DOWNLOAD_URL_DICT[dataset_name]
        print(f"Downloading {dataset_name} from {download_url} to {dest_dir}...")
        
        zip_path = os.path.join(dest_dir, f"{dataset_name}.zip")
        should_remove_zip = True

        try:
             if not download_file_from_url(download_url, zip_path):
                 print(f"Download failed for {dataset_name}.")
                 return
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return

    print(f"Extracting {dataset_name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    except zipfile.BadZipFile:
        print(f"Error: 压缩包不完整或已损坏 ({zip_path})！")
        if should_remove_zip:
            print("这可能是因为网络中断导致文件不全。正在删除损坏的文件...")
            os.remove(zip_path)
            print("请重新运行脚本，它会从头下载或触发正确的断点续传。")
        else:
            print("请检查您的本地源文件是否完整。")
        return
    except Exception as e:
        print(f"Error extracting {dataset_name}: {e}")
        return
    
    # Cleanup only if downloaded
    if should_remove_zip:
        os.remove(zip_path)
        
    print(f"Successfully installed {dataset_name} at {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OOD Datasets for TVI")
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory")
    parser.add_argument("--source_dir", type=str, default=None, help="Directory containing local dataset zip files")
    parser.add_argument("--datasets", nargs="+", default=["all"], 
                        choices=["places365", "inaturalist", "openimage_o", "texture", "all"])
    
    args = parser.parse_args()
    
    target_datasets = args.datasets
    if "all" in target_datasets:
        target_datasets = ["inaturalist", "openimage_o"]
        
    print(f"Target datasets: {target_datasets}")
    print(f"Data root: {os.path.abspath(args.data_root)}")
    if args.source_dir:
        print(f"Source directory: {os.path.abspath(args.source_dir)}")
    
    for ds in target_datasets:
        download_and_extract(ds, args.data_root, source_dir=args.source_dir)
