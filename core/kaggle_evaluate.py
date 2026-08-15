import os
import argparse
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix
)
import kagglehub

from core.config import AegisConfig
from utils.preprocessing import Preprocessor
from core.agent import ForensicAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Aegis-X on a Kaggle dataset.")
    parser.add_argument("--dataset", type=str, default="manjilkarki/deepfake-and-real-images", help="Kaggle dataset handle.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save results.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples to process per class.")
    return parser.parse_args()

def setup_results_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return Path(output_dir)

def download_dataset(dataset_handle):
    print(f"\n[🔄] Downloading dataset: {dataset_handle}...")
    path = kagglehub.dataset_download(dataset_handle)
    print(f"[✔] Dataset downloaded to: {path}")
    return Path(path)

def map_dataset_folders(dataset_root):
    """Identifies folders for real and fake samples."""
    print(f"\n[🔄] Scanning dataset structure in {dataset_root}...")
    
    real_dirs = []
    fake_dirs = []
    
    # Common folder name patterns
    real_patterns = ["real", "genuine", "authentic", "original"]
    fake_patterns = ["fake", "deepfake", "manipulated", "artificial", "ai"]
    
    for item in dataset_root.iterdir():
        if item.is_dir():
            name_lower = item.name.lower()
            if any(p in name_lower for p in real_patterns):
                real_dirs.append(item)
            elif any(p in name_lower for p in fake_patterns):
                fake_dirs.append(item)
    
    # Specific fix for "manjilkarki/deepfake-and-real-images"
    # User says: "two folders for real and one for fake"
    if not real_dirs or not fake_dirs:
        # Fallback to recursively finding folders named 'real' or 'fake'
        for root, dirs, files in os.walk(dataset_root):
            for d in dirs:
                d_lower = d.lower()
                if any(p in d_lower for p in real_patterns):
                    real_dirs.append(Path(root) / d)
                elif any(p in d_lower for p in fake_patterns):
                    fake_dirs.append(Path(root) / d)
                    
    # Remove duplicates
    real_dirs = list(set(real_dirs))
    fake_dirs = list(set(fake_dirs))
    
    print(f"    Found Real folders: {[d.name for d in real_dirs]}")
    print(f"    Found Fake folders: {[d.name for d in fake_dirs]}")
    
    return real_dirs, fake_dirs

def get_files(directories, max_samples=None):
    files = []
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.mp4", "*.avi", "*.mov"]
    
    for d in directories:
        count = 0
        for ext in extensions:
            for f in d.rglob(ext):
                files.append(f)
                count += 1
                if max_samples and count >= max_samples:
                    break
            if max_samples and count >= max_samples:
                break
    return files

def evaluate(dataset_root, real_dirs, fake_dirs, output_path, max_samples=None):
    real_files = get_files(real_dirs, max_samples)
    fake_files = get_files(fake_dirs, max_samples)
    
    all_files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
    
    if not all_files:
        print("Error: No media files found to evaluate.")
        return
        
    print(f"\n[📊] Starting evaluation on {len(all_files)} files...")
    
    config = AegisConfig()
    preprocessor = Preprocessor(config)
    
    results = []
    y_true = []
    y_scores = [] # probability of being fake
    y_pred = []
    
    for file_path, label in tqdm(all_files, desc="Processing Pipeline"):
        try:
            # 1. Preprocess
            prep_result = preprocessor.process_media(str(file_path))
            if not prep_result.has_face:
                continue
                
            # 2. Analyze
            agent = ForensicAgent(config)
            final_score = 0.0
            verdict = "REAL"
            
            # Generator loop
            for event in agent.analyze(prep_result, media_path=str(file_path)):
                if event.event_type == "verdict":
                    verdict = event.data.get("verdict", "REAL")
                    final_score = event.data.get("score", 0.0) # score is authenticity 0=fake, 1=real
            
            # predicted fake probability = 1.0 - authenticity score
            fake_prob = 1.0 - final_score
            predicted_label = 1 if verdict == "FAKE" else 0
            
            results.append({
                "filename": file_path.name,
                "true_label": label,
                "predicted_label": predicted_label,
                "fake_probability": fake_prob,
                "verdict": verdict
            })
            
            y_true.append(label)
            y_scores.append(fake_prob)
            y_pred.append(predicted_label)
            
        except Exception as e:
            # print(f"\nError processing {file_path.name}: {e}") # suppress logs for cleaner output
            continue

    if not y_true:
        print("Error: No samples were successfully processed.")
        return

    # --- METRICS CALCULATION ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # --- DESIGNED OUTPUT ---
    print("\n" + "="*50)
    print("         AEGIS-X EVALUATION REPORT")
    print("="*50)
    print(f"  Dataset:         manjilkarki/deepfake-and-real-images")
    print(f"  Total Processed: {len(y_true)}")
    print("-" * 50)
    print(f"  Accuracy:        {acc:>8.4f}")
    print(f"  Precision:       {prec:>8.4f}")
    print(f"  Recall (TPR):    {rec:>8.4f}")
    print(f"  F1-Score:        {f1:>8.4f}")
    print(f"  FPR:             {fpr:>8.4f}")
    print(f"  ROC AUC Score:   {auc:>8.4f}")
    print("="*50)
    
    # --- SAVE CSV ---
    df = pd.DataFrame(results)
    csv_path = output_path / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[💾] Detailed results saved to: {csv_path}")
    
    # --- ROC CURVE ---
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(10, 7), facecolor='#f8f9fa')
    plt.plot(fpr_arr, tpr_arr, color='#2563eb', lw=3, label=f'Aegis-X (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#94a3b8', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) - Aegis-X Forensic Evaluation', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = output_path / "roc_curve.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[📸] ROC curve visualization saved to: {plot_path}")

def main():
    args = parse_args()
    output_path = setup_results_dir(args.output_dir)
    
    # 1. Download
    dataset_root = download_dataset(args.dataset)
    
    # 2. Map Folders
    real_dirs, fake_dirs = map_dataset_folders(dataset_root)
    
    if not real_dirs or not fake_dirs:
        print("Error: Could not automatically detect real or fake folders.")
        return
        
    # 3. Evaluate
    evaluate(dataset_root, real_dirs, fake_dirs, output_path, args.max_samples)

if __name__ == "__main__":
    main()
