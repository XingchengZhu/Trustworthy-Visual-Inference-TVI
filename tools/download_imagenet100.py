import os
import sys

"""
# 1. 添加 Cloudflare GPG key
curl -fsSL https://pkg.cloudflareclient.com/pubkey.gpg | sudo gpg --yes --dearmor --output /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg

# 2. 添加源
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] https://pkg.cloudflareclient.com/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/cloudflare-client.list

# 3. 更新并安装
sudo apt-get update && sudo apt-get install cloudflare-warp
---
# 1. 注册设备
warp-cli registration new

# 2. 设置模式为本地代理 (默认端口 40000)
warp-cli mode proxy

# 3. 连接
warp-cli connect

# 4. 验证是否连接成功 (应该显示 Connected)
warp-cli status

"""

def download_imagenet100(target_dir="./data/imagenet100"):
    # Check if pysocks is installed (implicitly needed for socks proxy in requests)
    try:
        import socks
    except ImportError:
        print("[WARNING] 'pysocks' module not found. SOCKS proxy might fail.")
        print("Please run: pip install pysocks")

    # Check if kaggle is installed
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] 'kaggle' library not found. Please run 'pip install kaggle'.")
        return False

    # --- Proxy Settings (User Request) ---
    # Cloudflare WARP Proxy at 127.0.0.1:40000
    os.environ["http_proxy"] = "socks5://127.0.0.1:40000"
    os.environ["https_proxy"] = "socks5://127.0.0.1:40000"
    
    # --- Credentials (User Request) ---
    # Hardcoded as specifically requested for this environment
    os.environ['KAGGLE_USERNAME'] = "lemanchu"
    os.environ['KAGGLE_KEY'] = "a376c7f7c142e6f276aa839fca61c474"
    
    print("Initializing Kaggle API with Proxy Config...")
    print(f"HTTP_PROXY: {os.environ.get('http_proxy')}")
    print(f"HTTPS_PROXY: {os.environ.get('https_proxy')}")
    print(f"USER: {os.environ.get('KAGGLE_USERNAME')}")

    try:
        api = KaggleApi()
        api.authenticate()
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        print(f"Attempting to download ImageNet-100 to {target_dir}...")
        print("Downloading via Cloudflare WARP...")
        
        api.dataset_download_files(
            'ambityga/imagenet100',
            path=target_dir,
            unzip=True,
            quiet=False
        )
        print("[SUCCESS] ImageNet-100 downloaded and unzipped.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    download_imagenet100()
