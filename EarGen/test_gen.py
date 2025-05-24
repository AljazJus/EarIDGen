import torch
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import random
from torchvision import transforms
import torch.nn.functional as F
# Assuming EGD.py and params.py are in the same directory or accessible in PYTHONPATH
from EGD import EmbeddingGuidedDiffusion

import params as p # Import the parameters
import pandas as pd

import sys
sys.path.insert(0, '/home/aljazjustin/ear_generation')
from arcface_torch.backbones import get_model as get_ear_recognition_model # Alias to avoid conflict

# Define the transformation for the ear recognition model
# This should match what was used during training in fine_tune.py
EAR_MODEL_INPUT_SIZE = (112, 112) # Standard size for ArcFace models
ear_transform = transforms.Compose([
    transforms.Resize(EAR_MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes to [-1,1]
])

def load_ear_recognition_model(model_path, model_name, embedding_size, device):
    """Load the ear recognition model used for embedding calculation"""
    print(f"Loading ear recognition model from: {model_path}")
    try:
        ear_model = get_ear_recognition_model(
            model_name,  # e.g., 'vit_b' as used in fine_tune.py
            fp16=False,  # Typically False for inference
            num_features=embedding_size
        )
        ear_model.load_state_dict(torch.load(model_path, map_location=device))
        ear_model = ear_model.to(device)
        ear_model.eval()
        print("Ear recognition model loaded successfully.")
        return ear_model
    except Exception as e:
        print(f"Error loading ear recognition model: {e}")
        return None

def get_ear_embedding(pil_image, model, transform, device):
    """Extract embedding from an image using the ear recognition model"""
    if model is None:
        return None
        
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(image_tensor)
    
    return embedding

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    if embedding1 is None or embedding2 is None:
        return None
        
    # Normalize the embeddings (important for cosine similarity)
    embedding1_normalized = F.normalize(embedding1, p=2, dim=1)
    embedding2_normalized = F.normalize(embedding2, p=2, dim=1)
    
    # Calculate cosine similarity
    similarity = torch.sum(embedding1_normalized * embedding2_normalized, dim=1).item()
    return similarity

def load_model_with_finetuned_unet(finetuned_unet_path, config, device):
    """Loads the EGD model and then loads the fine-tuned UNet weights."""
    print(f"Initializing EGD model with base: {config.MODEL_ID}")
    egd_model = EmbeddingGuidedDiffusion(
        model_id=config.MODEL_ID,
        local_dir_base=config.LOCAL_MODEL_CACHE_DIR,
        embedding_dim=config.EMBEDDING_DIM,
        helper_prompt=config.HELPER_PROMPT if hasattr(config, 'HELPER_PROMPT') else "",
        device=device,
        num_feature_tokens=config.NUM_FEATURE_TOKENS if hasattr(config, 'NUM_FEATURE_TOKENS') else 1,
        repeat_feature_vector=config.REPEAT_FEATURE_VECTOR if hasattr(config, 'REPEAT_FEATURE_VECTOR') else 1,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)

    print(f"Looking for fine-tuned UNet weights in: {finetuned_unet_path}")

    # Define possible weight file names
    weights_name_safetensors_explicit = "diffusion_pytorch_model.safetensors"
    weights_name_safetensors_standard = "model.safetensors"
    weights_name_bin = "diffusion_pytorch_model.bin"

    # Construct full paths
    path_safetensors_explicit = os.path.join(finetuned_unet_path, weights_name_safetensors_explicit)
    path_safetensors_standard = os.path.join(finetuned_unet_path, weights_name_safetensors_standard)
    path_bin = os.path.join(finetuned_unet_path, weights_name_bin)

    actual_weights_path = None
    if os.path.exists(path_safetensors_explicit):
        actual_weights_path = path_safetensors_explicit
        print(f"Found weights file: {weights_name_safetensors_explicit}")
    elif os.path.exists(path_safetensors_standard):
        actual_weights_path = path_safetensors_standard
        print(f"Found weights file: {weights_name_safetensors_standard}")
    elif os.path.exists(path_bin):
        actual_weights_path = path_bin
        print(f"Found weights file: {weights_name_bin}")
    else:
        # Try looking directly for the filename with epoch
        if os.path.isfile(finetuned_unet_path):
            actual_weights_path = finetuned_unet_path
            print(f"Using weights file directly: {finetuned_unet_path}")
        else:
            raise FileNotFoundError(
                f"Could not find UNet weights ({weights_name_safetensors_explicit}, "
                f"{weights_name_safetensors_standard}, or {weights_name_bin}) "
                f"in {finetuned_unet_path}"
            )

    print(f"Attempting to load weights from: {actual_weights_path}")

    if actual_weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        try:
            state_dict = load_file(actual_weights_path, device=device.type)
            egd_model.unet.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading .safetensors file with safetensors.torch.load_file: {e}")
            print("Attempting fallback with torch.load for .safetensors...")
            state_dict = torch.load(actual_weights_path, map_location=device)
            if "state_dict" in state_dict: 
                state_dict = state_dict["state_dict"]
            egd_model.unet.load_state_dict(state_dict)
    else: # .bin file or other format
        try:
            state_dict = torch.load(actual_weights_path, map_location=device)
            # Check if it's a full checkpoint with a state_dict key
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # Check if weights have "module." prefix (from DataParallel)
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k in state_dict}
            egd_model.unet.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise

    egd_model.unet.eval()
    egd_model.vae.eval()
    egd_model.text_encoder.eval()
    print("Model loaded and fine-tuned UNet weights applied.")
    return egd_model

def main_test():
    # --- Configuration ---
    config = p

    # Get the run_name either from the command line or a default
    import argparse
    parser = argparse.ArgumentParser(description="Test ear generation model")
    parser.add_argument("--run_name", type=str, default="/home/aljazjustin/ear_generation/output_finetune_arc2face_style/lr1e-05_bs2_epochs50_emb768_idlossTrue",
                      help="Name of the saved model run")
    parser.add_argument("--epoch", type=str, default="unet_epoch_40",
                      help="Epoch checkpoint to test (e.g., unet_epoch_50)")
    parser.add_argument("--num_images", type=int, default=5,
                      help="Number of images to test")
    args = parser.parse_args()
    
    run_name = args.run_name
    epoch_to_test = args.epoch
    num_images_to_test = args.num_images
    
    finetuned_unet_path = os.path.join(config.OUTPUT_DIR_BASE, run_name, epoch_to_test)
    
    feature_vectors_json_path = config.FEATURE_VECTORS_PATH
    original_image_dir = config.IMAGE_DIR
    image_file_extension = config.IMAGE_FILE_EXTENSION
    image_size_for_display = 256

    # Output directory for saved images
    output_dir = os.path.join(config.OUTPUT_DIR_BASE, run_name, "test_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Test results will be stored in: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load EGD Model ---
    egd_model = load_model_with_finetuned_unet(finetuned_unet_path, config, device)

    # --- Load Ear Recognition Model ---
    ear_model = load_ear_recognition_model(
        model_path=config.EAR_MODEL_PATH,
        model_name=config.EAR_MODEL_NAME,
        embedding_size=config.EAR_EMBEDDING_SIZE,
        device=device
    )
    
    # --- Load Feature Vectors ---
    print(f"Loading feature vectors from: {feature_vectors_json_path}")
    with open(feature_vectors_json_path, 'r') as f:
        all_feature_vectors_data = json.load(f)
    
    all_image_ids = list(all_feature_vectors_data.keys())
    if len(all_image_ids) < num_images_to_test:
        print(f"Warning: Requested {num_images_to_test} images, but only {len(all_image_ids)} available in feature vectors JSON.")
        num_images_to_test = len(all_image_ids)

    # --- Select Images to Test ---
    if not all_image_ids:
        print("No image IDs found in the feature vector JSON. Exiting.")
        return
        
    selected_image_ids = random.sample(all_image_ids, num_images_to_test)
    print(f"Selected image IDs for testing: {selected_image_ids}")

    # --- Generate and Display ---
    fig, axes = plt.subplots(2, num_images_to_test, figsize=(num_images_to_test * 5, 10))

    # Handle the case when num_images_to_test is 1
    if num_images_to_test == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # Reshape for consistent indexing

    all_scores = []  # To store similarity scores (Original vs Generated)
    processed_ids = []  # IDs with successful processing

    for i, image_id in enumerate(selected_image_ids):
        print(f"\nProcessing image ID: {image_id}")

        # 1. Get Feature Vector
        feature_vector_np = np.array(all_feature_vectors_data[image_id], dtype=np.float32)
        feature_vector_tensor = torch.from_numpy(feature_vector_np).unsqueeze(0).to(device)

        # 2. Load Original Image
        original_image_path = None
        try:
            if '-' in image_id:
                folder_part, file_part_no_ext = image_id.split('-', 1)
                constructed_path = os.path.join(original_image_dir, folder_part, file_part_no_ext + image_file_extension)
                if os.path.exists(constructed_path):
                    original_image_path = constructed_path
            
            # Fallback to direct path if the folder structure doesn't match
            if original_image_path is None:
                direct_path = os.path.join(original_image_dir, image_id + image_file_extension)
                if os.path.exists(direct_path):
                    original_image_path = direct_path
            
            if original_image_path is None:
                raise FileNotFoundError(f"Could not find image for ID {image_id}")
                
            original_image_pil = Image.open(original_image_path).convert("RGB")
            print(f"Loaded original image: {original_image_path}")
        except FileNotFoundError:
            print(f"Error: Original image not found for {image_id}")
            # Update axis indexing for horizontal layout
            axes[0, i].set_title(f"Original: {image_id}\n(Not Found)", fontsize=24, fontweight='bold')
            axes[0, i].axis('off')
            axes[1, i].set_title(f"Generated: {image_id}\n(Skipped)", fontsize=24, fontweight='bold')
    
            axes[1, i].axis('off')
            continue  # Skip to next image_id

        # 3. Generate Image
        print("Generating image...")
        with torch.no_grad():
            generated_image_tensor = egd_model.generate_image(
                feature_vectors_batch=feature_vector_tensor,
                num_inference_steps=100, 
                guidance_scale=1.50,    
                height=config.IMAGE_SIZE,
                width=config.IMAGE_SIZE
            ) 
        generated_image_pil = transforms.ToPILImage()(generated_image_tensor.squeeze(0).cpu())
        print("Image generation complete.")

        # 4. Save individual images
        # original_save_path = os.path.join(output_dir, f"original_{image_id}.png")
        # generated_save_path = os.path.join(output_dir, f"generated_{image_id}.png")
        # original_image_pil.save(original_save_path)
        # generated_image_pil.save(generated_save_path)
        print(f"Saved original and generated images to {output_dir}")

        # 5. Calculate similarity using ear recognition model
        similarity_score = None
        if ear_model:
            try:
                original_embedding = get_ear_embedding(original_image_pil, ear_model, ear_transform, device)
                generated_embedding = get_ear_embedding(generated_image_pil, ear_model, ear_transform, device)
                
                if original_embedding is not None and generated_embedding is not None:
                    similarity_score = calculate_similarity(original_embedding, generated_embedding)
                    all_scores.append(similarity_score)
                    processed_ids.append(image_id)
                    print(f"Ear model similarity for {image_id}: {similarity_score:.4f}")
            except Exception as e:
                print(f"Error calculating similarity: {e}")
        
        # 6. Display images - update axis indexing for horizontal layout
        # Original Image - now in row 0, column i
        axes[0, i].imshow(original_image_pil)
        axes[0, i].set_title(f"Original: {image_id}", fontsize=24, fontweight='bold')
        axes[0, i].axis('off')

        # Generated Image - now in row 1, column i
        gen_title = f"Generated: {image_id}"
        if similarity_score is not None:
            gen_title += f"\nSimilarity: {similarity_score:.4f}"
        axes[1, i].imshow(generated_image_pil)
        axes[1, i].set_title(gen_title, fontsize=24, fontweight='bold')
        axes[1, i].axis('off')

    plt.tight_layout()
    
    # Save the comparison figure
    comparison_path = os.path.join(output_dir, "comparison.png")
    fig.savefig(comparison_path)
    print(f"Saved comparison image to: {comparison_path}")

    # Save similarity scores
    if all_scores:
        avg_score = np.mean(all_scores)
        print(f"\nAverage similarity score: {avg_score:.4f}")
        scores_data = {
            "image_ids": processed_ids,
            "scores": all_scores,
            "average_score": avg_score
        }
        scores_path = os.path.join(output_dir, "similarity_scores.json")
        with open(scores_path, 'w') as f:
            json.dump(scores_data, f, indent=4)
        print(f"Saved similarity scores to: {scores_path}")

if __name__ == "__main__":
    main_test()