import os 
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import argparse # For overriding params via command line

from EGD import EmbeddingGuidedDiffusion 
import params as p # Import the parameters

class EarGenDataset(Dataset):
    def __init__(self, feature_vectors_path, image_dir, image_size=(512, 512), image_file_extension=".png"):
        print(f"Loading feature vectors from: {feature_vectors_path}")
        with open(feature_vectors_path, 'r') as f:
            self.feature_vectors_data = json.load(f)
        
        self.image_dir = image_dir
        self.image_filenames_keys = list(self.feature_vectors_data.keys()) # These are keys like "390-22"
        self.image_file_extension = image_file_extension
        print(f"Found {len(self.image_filenames_keys)} feature vectors/image pairs.")
        print(f"Image directory set to: {self.image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),            
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # For RGB
        ])

    def __len__(self):
        return len(self.image_filenames_keys)

    def __getitem__(self, idx):
        # image_filename_base is the key from the JSON, e.g., "390-22"
        image_filename_key = self.image_filenames_keys[idx] 
        
        feature_vector = np.array(self.feature_vectors_data[image_filename_key], dtype=np.float32)
        feature_vector_tensor = torch.from_numpy(feature_vector)

        # --- Path Correction ---
        # Assuming image_filename_key is like "folder-file" e.g., "390-22"
        try:
            folder_part, file_part_no_ext = image_filename_key.split('-', 1)
        except ValueError:
            print(f"Error: Could not parse image key '{image_filename_key}' into folder-file format.")
            # Fallback or raise error: if keys are sometimes just filenames without folder
            # For now, let's assume it's always folder-file
            # If not, you might need a more robust way to map keys to paths
            # e.g. if key is "image123.png" and it's in "folderA/image123.png"
            # then the JSON key itself should probably be "folderA-image123" or similar
            # or your JSON needs to store the full relative path.
            # For "390/22.png" structure, the key "390-22" is good.
            raise ValueError(f"Image key '{image_filename_key}' is not in the expected 'folder-file' format.")

        image_path = os.path.join(self.image_dir, folder_part, file_part_no_ext + self.image_file_extension)
        # --- End Path Correction ---
        
        try:
            image = Image.open(image_path).convert("RGB") 
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path} (derived from key: {image_filename_key})")
            # You might want to return None or a placeholder, or skip this item
            # For now, re-raising the error is fine for debugging.
            raise 
            
        image_tensor = self.transform(image)
        
        # Return the original key as 'id' if needed for tracking
        return {'feature_vector': feature_vector_tensor, 'image_tensor': image_tensor, 'id': image_filename_key}