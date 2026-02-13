import os
import argparse
import gdown
import zipfile

# OpenOOD Benchmark IDs
DOWNLOAD_ID_DICT = {
    'places365': '1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
}

# Target Directories relative to data/
DIR_DICT = {
    'places365': 'images_classic/places365',
    'texture': 'images_classic/texture',
    'inaturalist': 'images_largescale/inaturalist',
    'openimage_o': 'images_largescale/openimage_o',
}

def download_and_extract(dataset_name, data_root):
    if dataset_name not in DOWNLOAD_ID_DICT:
        print(f"Dataset {dataset_name} not supported.")
        return

    rel_path = DIR_DICT[dataset_name]
    # OpenOOD script downloads to parent dir then extracts.
    # e.g. images_classic/places365
    # The zip usually contains the folder structure or just images?
    # OpenOOD logic: store_path = ...; gdown(output=store_path); extract;
    
    # We'll stick to a robust approach:
    # 1. Ensure parent dir exists
    dest_dir = os.path.join(data_root, rel_path)
    if os.path.exists(dest_dir) and os.listdir(dest_dir):
        print(f"Dataset {dataset_name} already exists at {dest_dir}. Skipping.")
        return

    os.makedirs(dest_dir, exist_ok=True)
    
    file_id = DOWNLOAD_ID_DICT[dataset_name]
    print(f"Downloading {dataset_name} (ID: {file_id}) to {dest_dir}...")
    
    # gdown output path. If it's a folder, gdown might download a file inside it.
    # We want the zip file.
    zip_path = os.path.join(dest_dir, f"{dataset_name}.zip")

    proxy_domain = "gdrive.testx.asia" 
    download_url = f"https://{proxy_domain}/uc?id={file_id}"
    
    try:
        gdown.download(url=download_url, output=zip_path, quiet=False, resume=True)
        
        if not os.path.exists(zip_path):
            print(f"Download failed for {dataset_name}.")
            return

        print(f"Extracting {dataset_name}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
        except zipfile.BadZipFile:
            print(f"Error: 压缩包不完整或已损坏 ({dataset_name}.zip)！")
            print("这可能是因为网络中断导致文件不全。正在删除损坏的文件...")
            os.remove(zip_path)
            print("请重新运行脚本，它会从头下载或触发正确的断点续传。")
            return
        
        # Cleanup
        os.remove(zip_path)
        print(f"Successfully installed {dataset_name} at {dest_dir}")
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OOD Datasets for TVI")
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory")
    parser.add_argument("--datasets", nargs="+", default=["all"], 
                        choices=["places365", "inaturalist", "openimage_o", "texture", "all"])
    
    args = parser.parse_args()
    
    target_datasets = args.datasets
    if "all" in target_datasets:
        target_datasets = ["places365", "inaturalist", "openimage_o", "texture"]
        
    print(f"Target datasets: {target_datasets}")
    print(f"Data root: {os.path.abspath(args.data_root)}")
    
    for ds in target_datasets:
        download_and_extract(ds, args.data_root)
