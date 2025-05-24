import torch
import torch.nn.functional as F
from PIL import Image
from siamese import SiameseNetwork  # Import your model
import os
import torchvision.transforms as transforms
import json # Added for JSON operations

import sys
sys.path.insert(0, '/home/aljazjustin/ear_generation')
# Import the get_model function from the ArcFace backbones
from arcface_torch.backbones import get_model

weights_path = '/home/aljazjustin/ear_generation/models_embeding/vit16_corrected_margin_epoh7/model.pt'
model_name = 'vit_b'
embedding_size = 768 # Ensure this matches your training config for vit16
fp16_inference = True # Set to True if your model was trained with fp16 and you want to use it
# --- Device Configuration ---

# --- Load Model ---


# Define the base path for image folders
image_folders_base_path = '/home/aljazjustin/ear_generation/data/SOURCES/IBB/ears-by-fols'
output_json_path = '/home/aljazjustin/ear_generation/data/feature_vectors_arc.json'


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# device will be defined in __main__
device = None

# Function to process an image
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) # device will be set in main
    return image_tensor

def img2vec(image_path, model_instance):
    # This function can now use the passed model_instance
    image_tensor = process_image(image_path)
    with torch.no_grad():
        vector = model_instance(image_tensor)
    return vector.squeeze().cpu().numpy() # Return as numpy array or list

        
if __name__ == "__main__":
    # Define device here so process_image can access it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading ArcFace model...")
    model = get_model(model_name, fp16=fp16_inference, num_features=embedding_size)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("ArcFace model loaded successfully.")

    all_feature_vectors = {}

    # List subdirectories
    try:
        subfolders = sorted([f.name for f in os.scandir(image_folders_base_path) if f.is_dir()])
    except FileNotFoundError:
        print(f"Error: Base directory not found: {image_folders_base_path}")
        exit()

    folders_processed_count = 0
    for folder_name in subfolders:
        # if folders_processed_count >= 10:
        #     print("Processed the first 10 folders. Stopping.")
        #     break
        
        current_folder_path = os.path.join(image_folders_base_path, folder_name)
        print(f"Processing folder: {current_folder_path}")
        
        image_files = [
            f for f in os.listdir(current_folder_path) 
            if os.path.isfile(os.path.join(current_folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not image_files:
            print(f"No images found in {current_folder_path}. Skipping.")
            continue

        for image_name in image_files:
            image_path = os.path.join(current_folder_path, image_name)
            try:
                # Process image and get feature vector
                img_tensor = process_image(image_path)
                with torch.no_grad():
                    feature_vector_tensor = model(img_tensor)
                
                # Convert tensor to list for JSON serialization
                # Squeeze to remove batch dimension if present, then convert to CPU and list
                feature_vector_list = feature_vector_tensor.squeeze().cpu().tolist()
                
                # Create a unique key: folder_name-image_name
                # Remove extension from image_name for cleaner key if desired
                image_name_without_ext = os.path.splitext(image_name)[0]
                storage_key = f"{folder_name}-{image_name_without_ext}"
                
                all_feature_vectors[storage_key] = feature_vector_list
                # print(f"  Processed and stored vector for: {storage_key}")

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        
        folders_processed_count += 1
        print(f"Finished processing folder: {folder_name}. Total folders processed: {folders_processed_count}")


    # Save all feature vectors to a JSON file
    try:
        with open(output_json_path, 'w') as f:
            json.dump(all_feature_vectors, f, indent=4)
        print(f"All feature vectors saved to {output_json_path}")
    except IOError:
        print(f"Error: Could not write to JSON file at {output_json_path}")
    except Exception as e:
        print(f"An unexpected error occurred while saving JSON: {e}")
