import os
from pathlib import Path
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Please install the datasets library first: pip install datasets")
    exit(1)

def setup_dataset(samples_per_class=500):
    dataset_name = "sadanandkaji/deepfake-image-dataset"
    print(f"Downloading {dataset_name} from Hugging Face...")
    
    # Load dataset
    ds = load_dataset(dataset_name, split="train", streaming=True)
    
    # Create dataset directories
    base_dir = Path("my_image_dataset")
    real_dir = base_dir / "real"
    fake_dir = base_dir / "fake"
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    real_count = 0
    fake_count = 0
    
    try:
        # Assuming typical datasets have 'image' and 'label' or similar
        print(f"Extracting {samples_per_class} real and {samples_per_class} fake images...")
        
        for idx, item in tqdm(enumerate(ds)):
            # Handle different label column names in HF datasets
            label = item.get('label', item.get('labels', item.get('is_fake', None)))
            img = item.get('image', item.get('img', None))
            
            if img is None or label is None:
                continue
                
            # Usually 0 is real, 1 is fake, though it can vary (e.g., 'Real'/'Fake')
            if isinstance(label, str):
                is_fake = label.lower() in ['fake', 'ai', '1', 'manipulated']
            else:
                is_fake = bool(label)
                
            if is_fake and fake_count < samples_per_class:
                # Save fake image
                img_path = fake_dir / f"fake_{fake_count}.jpg"
                img.convert('RGB').save(img_path)
                fake_count += 1
            elif not is_fake and real_count < samples_per_class:
                # Save real image
                img_path = real_dir / f"real_{real_count}.jpg"
                img.convert('RGB').save(img_path)
                real_count += 1
                
            if real_count >= samples_per_class and fake_count >= samples_per_class:
                break
                
        print("\nDataset generation complete!")
        print(f"Saved {real_count} real images to {real_dir}")
        print(f"Saved {fake_count} fake images to {fake_dir}")
        print("\nYou can now run the evaluation script with it:")
        print("python evaluate_pipeline.py --dataset_root my_image_dataset --real_dir real --fake_dir fake")
        
    except Exception as e:
        print(f"An error occurred while building the dataset: {str(e)}")

if __name__ == "__main__":
    setup_dataset()
