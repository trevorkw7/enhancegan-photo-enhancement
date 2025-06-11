import os
import random
from PIL import Image
import multiprocessing as mp
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ImageProcessor:
    def __init__(self, size=128):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size), Image.LANCZOS),
            transforms.ToTensor()
        ])
        
    def process_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

def save_tensor_as_image(tensor, save_path):
    """Save tensor as image with optimized settings"""
    try:
        img = transforms.ToPILImage()(tensor)
        img.save(save_path, quality=95, optimize=True)
        return True
    except Exception as e:
        print(f"Error saving {save_path}: {e}")
        return False

def prepare_dped_data(raw_dir, edited_dir, output_dir, size=128, train_frac=0.8, val_frac=0.1, batch_size=32):
    """Prepare DPED data with proper train/val/test splits using batched processing"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        for t in ['raw', 'edited']:
            (output_dir / split / t).mkdir(parents=True, exist_ok=True)

    # Find matching pairs
    raw_dir = Path(raw_dir)
    edited_dir = Path(edited_dir)
    
    raw_files = {f.stem: f for f in raw_dir.glob('*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    edit_files = {f.stem: f for f in edited_dir.glob('*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    
    common_stems = set(raw_files.keys()) & set(edit_files.keys())
    
    if not common_stems:
        print("ERROR: No matching files found!")
        return False
    
    print(f"Found {len(common_stems)} matching file pairs")
    
    # Create pairs and shuffle
    pairs = [(raw_files[stem], edit_files[stem]) for stem in common_stems]
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    print(f"Splitting {n} pairs: train={n_train}, val={n_val}, test={n-n_train-n_val}")
    
    splits = [
        ('train', pairs[:n_train]),
        ('val', pairs[n_train:n_train+n_val]),
        ('test', pairs[n_train+n_val:])
    ]

    processor = ImageProcessor(size=size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for split_name, items in splits:
        print(f"Processing {split_name}: {len(items)} pairs")
        
        # Process in batches
        for i in tqdm(range(0, len(items), batch_size), desc=f"Processing {split_name}"):
            batch = items[i:i + batch_size]
            
            # Process raw images
            raw_tensors = []
            raw_paths = []
            for raw_path, _ in batch:
                tensor = processor.process_image(raw_path)
                if tensor is not None:
                    raw_tensors.append(tensor)
                    raw_paths.append(output_dir / split_name / 'raw' / raw_path.name)
            
            # Process edited images
            edit_tensors = []
            edit_paths = []
            for _, edit_path in batch:
                tensor = processor.process_image(edit_path)
                if tensor is not None:
                    edit_tensors.append(tensor)
                    edit_paths.append(output_dir / split_name / 'edited' / edit_path.name)
            
            # Save processed images
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                # Save raw images
                executor.map(save_tensor_as_image, raw_tensors, raw_paths)
                # Save edited images
                executor.map(save_tensor_as_image, edit_tensors, edit_paths)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare DPED dataset')
    parser.add_argument('--raw_dir', type=str, required=True, help='Directory containing raw images')
    parser.add_argument('--edited_dir', type=str, required=True, help='Directory containing edited images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--size', type=int, default=128, help='Size to resize images to')
    parser.add_argument('--train_frac', type=float, default=0.9, help='Fraction of data for training')
    parser.add_argument('--val_frac', type=float, default=0.05, help='Fraction of data for validation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    
    args = parser.parse_args()
    
    print("🔄 Running data_prep.py...")
    success = prepare_dped_data(
        args.raw_dir,
        args.edited_dir,
        args.output_dir,
        args.size,
        args.train_frac,
        args.val_frac,
        args.batch_size
    )
    
    if success:
        print("\nData splits created:")
        for split in ['train', 'val', 'test']:
            raw_count = len(list(Path(args.output_dir) / split / 'raw').glob('*'))
            edited_count = len(list(Path(args.output_dir) / split / 'edited').glob('*'))
            print(f"   {split}: {raw_count} raw, {edited_count} edited")
        
        print("\nData preparation complete!")
    else:
        print("\nData preparation failed!") 