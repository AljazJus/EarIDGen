import torch
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import random
from torchvision import transforms
import torch.nn.functional as F
from collections import defaultdict
import itertools
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

def extract_class_from_image_id(image_id):
    """Extract class identifier from image ID"""
    if '-' in image_id:
        return image_id.split('-')[0]
    return image_id.split('_')[0] if '_' in image_id else image_id[:3]

def organize_images_by_class(all_feature_vectors_data):
    """Organize images by class and select samples"""
    classes_dict = defaultdict(list)
    
    for image_id in all_feature_vectors_data.keys():
        class_id = extract_class_from_image_id(image_id)
        classes_dict[class_id].append(image_id)
    
    return classes_dict

def select_test_images(classes_dict, num_classes=20, images_per_class=10):
    """Select test images: 3 for generation, 3 for comparison only"""
    selected_classes = random.sample(list(classes_dict.keys()), min(num_classes, len(classes_dict)))
    
    test_data = {}
    for class_id in selected_classes:
        class_images = classes_dict[class_id]
        if len(class_images) >= images_per_class:
            selected_images = random.sample(class_images, images_per_class)
            test_data[class_id] = {
                'generation_images': selected_images[:5],  # First 3 for generation
                'comparison_images': selected_images[5:]   # Last 3 for comparison only
            }
        else:
            print(f"Warning: Class {class_id} has only {len(class_images)} images, skipping.")
    
    return test_data

def generate_images_for_class(egd_model, feature_vectors_data, image_ids, config, device):
    """Generate images for a list of image IDs"""
    generated_images = {}
    
    for image_id in image_ids:
        print(f"Generating image for: {image_id}")
        feature_vector_np = np.array(feature_vectors_data[image_id], dtype=np.float32)
        feature_vector_tensor = torch.from_numpy(feature_vector_np).unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated_image_tensor = egd_model.generate_image(
                feature_vectors_batch=feature_vector_tensor,
                num_inference_steps=100, 
                guidance_scale=1.50,    
                height=config.IMAGE_SIZE,
                width=config.IMAGE_SIZE
            )
        generated_image_pil = transforms.ToPILImage()(generated_image_tensor.squeeze(0).cpu())
        generated_images[image_id] = generated_image_pil
    
    return generated_images

def load_original_images(image_ids, original_image_dir, image_file_extension):
    """Load original images for comparison"""
    original_images = {}
    
    for image_id in image_ids:
        try:
            original_image_path = None
            if '-' in image_id:
                folder_part, file_part_no_ext = image_id.split('-', 1)
                constructed_path = os.path.join(original_image_dir, folder_part, file_part_no_ext + image_file_extension)
                if os.path.exists(constructed_path):
                    original_image_path = constructed_path
            
            if original_image_path is None:
                direct_path = os.path.join(original_image_dir, image_id + image_file_extension)
                if os.path.exists(direct_path):
                    original_image_path = direct_path
            
            if original_image_path:
                original_image_pil = Image.open(original_image_path).convert("RGB")
                original_images[image_id] = original_image_pil
                print(f"Loaded original image: {image_id}")
            else:
                print(f"Warning: Could not find original image for {image_id}")
        except Exception as e:
            print(f"Error loading original image {image_id}: {e}")
    
    return original_images

def calculate_all_similarities(generated_images, original_images, test_data, ear_model, ear_transform, device):
    """Calculate all required similarity metrics"""
    results = {
        'intra_class_generated_vs_original': {},
        'intra_class_generated_vs_generated': {},
        'intra_class_original_vs_original': {},
        'inter_class_generated_vs_original': {},
        'inter_class_generated_vs_generated': {},
        'inter_class_original_vs_original': {}
    }
    
    all_classes = list(test_data.keys())
    
    for class_id in all_classes:
        print(f"\nProcessing similarities for class: {class_id}")
        
        # Get images for current class
        gen_ids = test_data[class_id]['generation_images']
        comp_ids = test_data[class_id]['comparison_images']
        all_class_ids = gen_ids + comp_ids
        
        class_generated = {img_id: generated_images[img_id] for img_id in gen_ids if img_id in generated_images}
        class_original = {img_id: original_images[img_id] for img_id in all_class_ids if img_id in original_images}
        
        # 1. Intra-class: Generated vs Original (same class)
        intra_gen_orig = []
        for gen_id in class_generated:
            gen_emb = get_ear_embedding(class_generated[gen_id], ear_model, ear_transform, device)
            for orig_id in class_original:
                orig_emb = get_ear_embedding(class_original[orig_id], ear_model, ear_transform, device)
                sim = calculate_similarity(gen_emb, orig_emb)
                if sim is not None:
                    intra_gen_orig.append(sim)
        
        # 2. Intra-class: Generated vs Generated (same class)
        intra_gen_gen = []
        gen_ids_list = list(class_generated.keys())
        for i, j in itertools.combinations(gen_ids_list, 2):
            gen_emb1 = get_ear_embedding(class_generated[i], ear_model, ear_transform, device)
            gen_emb2 = get_ear_embedding(class_generated[j], ear_model, ear_transform, device)
            sim = calculate_similarity(gen_emb1, gen_emb2)
            if sim is not None:
                intra_gen_gen.append(sim)
        
        # 3. Intra-class: Original vs Original (same class)
        intra_orig_orig = []
        orig_ids_list = list(class_original.keys())
        for i, j in itertools.combinations(orig_ids_list, 2):
            orig_emb1 = get_ear_embedding(class_original[i], ear_model, ear_transform, device)
            orig_emb2 = get_ear_embedding(class_original[j], ear_model, ear_transform, device)
            sim = calculate_similarity(orig_emb1, orig_emb2)
            if sim is not None:
                intra_orig_orig.append(sim)
        
        # Store intra-class results
        results['intra_class_generated_vs_original'][class_id] = intra_gen_orig
        results['intra_class_generated_vs_generated'][class_id] = intra_gen_gen
        results['intra_class_original_vs_original'][class_id] = intra_orig_orig
        
        # 4-6. Inter-class comparisons
        inter_gen_orig = []
        inter_gen_gen = []
        inter_orig_orig = []
        
        for other_class_id in all_classes:
            if other_class_id != class_id:
                other_gen_ids = test_data[other_class_id]['generation_images']
                other_comp_ids = test_data[other_class_id]['comparison_images']
                other_all_ids = other_gen_ids + other_comp_ids
                
                other_generated = {img_id: generated_images[img_id] for img_id in other_gen_ids if img_id in generated_images}
                other_original = {img_id: original_images[img_id] for img_id in other_all_ids if img_id in original_images}
                
                # Inter-class: Current class generated vs Other class original
                for gen_id in class_generated:
                    gen_emb = get_ear_embedding(class_generated[gen_id], ear_model, ear_transform, device)
                    # Sample random images from other class to avoid too many comparisons
                    sampled_other_orig = random.sample(list(other_original.keys()), min(3, len(other_original)))
                    for orig_id in sampled_other_orig:
                        orig_emb = get_ear_embedding(other_original[orig_id], ear_model, ear_transform, device)
                        sim = calculate_similarity(gen_emb, orig_emb)
                        if sim is not None:
                            inter_gen_orig.append(sim)
                
                # Inter-class: Current class generated vs Other class generated
                for gen_id in class_generated:
                    gen_emb = get_ear_embedding(class_generated[gen_id], ear_model, ear_transform, device)
                    for other_gen_id in other_generated:
                        other_gen_emb = get_ear_embedding(other_generated[other_gen_id], ear_model, ear_transform, device)
                        sim = calculate_similarity(gen_emb, other_gen_emb)
                        if sim is not None:
                            inter_gen_gen.append(sim)
                
                # Inter-class: Current class original vs Other class original
                sampled_class_orig = random.sample(list(class_original.keys()), min(3, len(class_original)))
                sampled_other_orig = random.sample(list(other_original.keys()), min(3, len(other_original)))
                for orig_id in sampled_class_orig:
                    orig_emb = get_ear_embedding(class_original[orig_id], ear_model, ear_transform, device)
                    for other_orig_id in sampled_other_orig:
                        other_orig_emb = get_ear_embedding(other_original[other_orig_id], ear_model, ear_transform, device)
                        sim = calculate_similarity(orig_emb, other_orig_emb)
                        if sim is not None:
                            inter_orig_orig.append(sim)
        
        # Store inter-class results
        results['inter_class_generated_vs_original'][class_id] = inter_gen_orig
        results['inter_class_generated_vs_generated'][class_id] = inter_gen_gen
        results['inter_class_original_vs_original'][class_id] = inter_orig_orig
    
    return results

def compute_statistics(results):
    """Compute statistics from similarity results"""
    statistics = {}
    
    for metric_type, class_results in results.items():
        all_scores = []
        class_stats = {}
        
        for class_id, scores in class_results.items():
            if scores:
                class_mean = np.mean(scores)
                class_std = np.std(scores)
                class_stats[class_id] = {
                    'mean': float(class_mean),
                    'std': float(class_std),
                    'count': len(scores),
                    'scores': scores
                }
                all_scores.extend(scores)
        
        if all_scores:
            overall_mean = np.mean(all_scores)
            overall_std = np.std(all_scores)
            statistics[metric_type] = {
                'overall_mean': float(overall_mean),
                'overall_std': float(overall_std),
                'overall_count': len(all_scores),
                'class_statistics': class_stats
            }
        else:
            statistics[metric_type] = {
                'overall_mean': 0.0,
                'overall_std': 0.0,
                'overall_count': 0,
                'class_statistics': {}
            }
    
    return statistics

def main_test():
    # --- Configuration ---
    config = p

    # Get the run_name either from the command line or a default
    import argparse
    parser = argparse.ArgumentParser(description="Test ear generation model with comprehensive similarity analysis")
    parser.add_argument("--run_name", type=str, default="/home/aljazjustin/ear_generation/output_finetune_arc2face_style/lr1e-05_bs2_epochs50_emb768_idlossTrue",
                      help="Name of the saved model run")
    parser.add_argument("--epoch", type=str, default="unet_epoch_40",
                      help="Epoch checkpoint to test (e.g., unet_epoch_50)")
    parser.add_argument("--num_classes", type=int, default=10,
                      help="Number of classes to test")
    args = parser.parse_args()
    
    run_name = args.run_name
    epoch_to_test = args.epoch
    num_classes = args.num_classes
    
    finetuned_unet_path = os.path.join(config.OUTPUT_DIR_BASE, run_name, epoch_to_test)
    
    feature_vectors_json_path = config.FEATURE_VECTORS_PATH
    original_image_dir = config.IMAGE_DIR
    image_file_extension = config.IMAGE_FILE_EXTENSION

    # Output directory for results
    output_dir = os.path.join(config.OUTPUT_DIR_BASE, run_name, "comprehensive_test_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Test results will be stored in: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Models ---
    egd_model = load_model_with_finetuned_unet(finetuned_unet_path, config, device)
    ear_model = load_ear_recognition_model(
        model_path=config.EAR_MODEL_PATH,
        model_name=config.EAR_MODEL_NAME,
        embedding_size=config.EAR_EMBEDDING_SIZE,
        device=device
    )
    
    if ear_model is None:
        print("Error: Could not load ear recognition model. Exiting.")
        return

    # --- Load and Organize Data ---
    print(f"Loading feature vectors from: {feature_vectors_json_path}")
    with open(feature_vectors_json_path, 'r') as f:
        all_feature_vectors_data = json.load(f)
    
    # Organize images by class
    classes_dict = organize_images_by_class(all_feature_vectors_data)
    print(f"Found {len(classes_dict)} classes")
    
    # Select test images
    test_data = select_test_images(classes_dict, num_classes=num_classes, images_per_class=6)
    print(f"Selected {len(test_data)} classes for testing")
    
    # --- Generate Images ---
    print("\n=== Generating Images ===")
    all_generated_images = {}
    for class_id, class_data in test_data.items():
        print(f"Generating images for class: {class_id}")
        generated_images = generate_images_for_class(
            egd_model, all_feature_vectors_data, 
            class_data['generation_images'], config, device
        )
        all_generated_images.update(generated_images)
    
    # --- Load Original Images ---
    print("\n=== Loading Original Images ===")
    all_image_ids = []
    for class_data in test_data.values():
        all_image_ids.extend(class_data['generation_images'])
        all_image_ids.extend(class_data['comparison_images'])
    
    all_original_images = load_original_images(all_image_ids, original_image_dir, image_file_extension)
    
    # --- Calculate Similarities ---
    print("\n=== Calculating Similarities ===")
    similarity_results = calculate_all_similarities(
        all_generated_images, all_original_images, test_data, 
        ear_model, ear_transform, device
    )
    
    # --- Compute Statistics ---
    print("\n=== Computing Statistics ===")
    statistics = compute_statistics(similarity_results)
    
    # --- Save Results ---
    results_data = {
        "test_configuration": {
            "run_name": run_name,
            "epoch": epoch_to_test,
            "num_classes": len(test_data),
            "classes_tested": list(test_data.keys())
        },
        "similarity_results": similarity_results,
        "statistics": statistics
    }
    
    results_path = os.path.join(output_dir, "comprehensive_similarity_analysis.json")
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"Saved comprehensive results to: {results_path}")
    
    # --- Print Summary ---
    print("\n=== SUMMARY ===")
    for metric_type, stats in statistics.items():
        print(f"{metric_type.replace('_', ' ').title()}: {stats['overall_mean']:.4f} ± {stats['overall_std']:.4f} (n={stats['overall_count']})")

if __name__ == "__main__":
    main_test()