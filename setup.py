import os
import sys
import shutil
import subprocess
from pathlib import Path

def run_cmd(cmd, desc):
    print(f"\n[🔄] {desc}...")
    try:
        subprocess.run(cmd, check=True, shell=isinstance(cmd, str))
        print(f"[✔] Successfully finished: {desc}")
    except subprocess.CalledProcessError as e:
        print(f"[✖] Failed: {desc}")
        print(f"Error details: {e}")
        sys.exit(1)

def setup_environment():
    print("="*60)
    print("          TRUTHSCOPE V2 INSTALLATION SCRIPT")
    print("="*60)

    # 1. Setup Virtual Environments
    python_exe = sys.executable

    # Main Env
    venv_bin = "Scripts" if os.name == 'nt' else "bin"
    python_cmd_main = os.path.join(".venv_main", venv_bin, "python")
    python_cmd_gpu = os.path.join(".venv_gpu", venv_bin, "python")

    if not os.path.exists(".venv_main"):
        run_cmd([python_exe, "-m", "venv", ".venv_main"], "Creating .venv_main")
    
    run_cmd(
        f"{python_cmd_main} -m pip install --upgrade pip && {python_cmd_main} -m pip install -r requirements-main.txt",
        "Installing main dependencies"
    )

    # GPU Env
    if not os.path.exists(".venv_gpu"):
        run_cmd([python_exe, "-m", "venv", ".venv_gpu"], "Creating .venv_gpu")
    
    run_cmd(
        f"{python_cmd_gpu} -m pip install --upgrade pip && {python_cmd_gpu} -m pip install -r requirements-gpu.txt",
        "Installing GPU dependencies"
    )

    # 2. Kaggle Weights Download
    print("\n" + "="*60)
    print("               MODEL WEIGHTS SETUP")
    print("="*60)
    
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("[!] ERROR: Kaggle API credentials not found!")
        print("    Please go to https://kaggle.com -> Account -> 'Create New API Token'")
        print("    Put the downloaded 'kaggle.json' inside ~/.kaggle/kaggle.json")
        if os.name != 'nt':
            print("    Then set permissions: chmod 600 ~/.kaggle/kaggle.json")
        else:
            print("    Then ensure only your user has access to ~/.kaggle/kaggle.json")
        print("    After setting that up, re-run this setup script.")
        sys.exit(1)

    # Install kaggle CLI temporarily into main venv to fetch the datasets
    run_cmd(f"{python_cmd_main} -m pip install kaggle", "Installing Kaggle CLI tool")

    download_dir = Path("downloads/raw_weights")
    download_dir.mkdir(parents=True, exist_ok=True)

    # Download from user's Kaggle dataset
    dataset_url = "gauravkumarjangid/aegis-pth"
    run_cmd(
        f"{python_cmd_main} -m kaggle datasets download -d {dataset_url} -p {download_dir} --unzip",
        f"Downloading and unzipping models from {dataset_url}"
    )

    # 3. Distribute Weights
    print("\n[🔄] Distributing weights to the unified models registry...")

    distributions = {
        "cnndetect_resnet50.pth": "models/freqnet/cnndetect_resnet50.pth",
        "efficientnet_b4.pth": "models/sbi/efficientnet_b4.pth",
        "xception_deepfake.pth": "models/xception/xception_deepfake.pth",
        "probe.pth": "models/univfd/probe.pth"
    }

    for src_file, dest_path in distributions.items():
        src_path = download_dir / src_file
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if src_path.exists():
            shutil.move(str(src_path), dest_path)
            print(f"  [✔] Installed: {dest_path}")
        else:
            print(f"  [!] Missing from Kaggle zip: {src_file}")

    # Cleanup temporary downloads
    shutil.rmtree(download_dir, ignore_errors=True)

    print("\n" + "="*60)
    print(" INSTALLATION COMPLETE! YOU ARE READY TO DEPLOY TRUTHSCOPE.")
    activate_cmd = ".venv_main\\Scripts\\activate" if os.name == 'nt' else "source .venv_main/bin/activate"
    print(f" Use: '{activate_cmd}' followed by 'python run_web.py' to start the server.")
    print("="*60)

if __name__ == "__main__":
    setup_environment()
