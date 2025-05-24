import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np
from tqdm.auto import tqdm
import argparse 
from torchvision.transforms import functional as TF

from EGD import EmbeddingGuidedDiffusion 
import params as p
from EarGenDataset import EarGenDataset

# Import for your ear recognition model
import sys
sys.path.insert(0, '/home/aljazjustin/ear_generation')
from arcface_torch.backbones import get_model as get_ear_recognition_model # Alias to avoid conflict

# Define the transformation for your ear recognition model (as in test_working_arcear2.py)
# This will be used in identity_consistency_loss
EAR_MODEL_INPUT_SIZE = (112, 112) # Or your model's specific input size
ear_recognition_transform_to_tensor_and_norm = transforms.Compose([
    # Resize is handled separately on tensors, this is for PIL if needed
    # transforms.Resize(EAR_MODEL_INPUT_SIZE), 
    # transforms.ToTensor(), # Converts PIL [0,255] to Tensor [0,1] or keeps tensor as is
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes Tensor [0,1] to [-1,1]
])


def identity_consistency_loss(egd_model_instance, generated_images, target_embeddings, weight=1.0):
    """
    Calculate identity consistency loss.
    egd_model_instance: The instance of EmbeddingGuidedDiffusion, expected to have .ear_recognition_model
    generated_images: Tensor of generated images, typically in range [-1, 1] from VAE.
    target_embeddings: Ground truth ear embeddings.
    """
    batch_size = generated_images.shape[0]
    
    if egd_model_instance.ear_recognition_model is None:
        # Fallback to basic stats if ear model not available (should not happen in proper setup)
        print("Warning: ear_recognition_model not found in egd_model_instance. Using placeholder for identity loss.")
        processed_images_for_loss = (generated_images.detach() + 1.0) / 2.0 # Normalize to [0, 1]
        processed_images_for_loss = torch.clamp(processed_images_for_loss, 0.0, 1.0)
        processed_images_for_loss = TF.resize(processed_images_for_loss, [100,100], antialias=True) # Fallback resize
        
        generated_embeddings_placeholder = torch.cat([
            processed_images_for_loss.view(batch_size, -1).mean(dim=1, keepdim=True),
            processed_images_for_loss.view(batch_size, -1).std(dim=1, keepdim=True)
        ], dim=1)
        generated_embeddings_placeholder = F.normalize(generated_embeddings_placeholder, p=2, dim=1)
        generated_embeddings = F.pad(
            generated_embeddings_placeholder, 
            (0, target_embeddings.shape[1] - generated_embeddings_placeholder.shape[1]),
            "constant", 0
        )
    else:
        # Process generated images for the ear recognition model
        # 1. Normalize generated_images from [-1, 1] (VAE output) to [0, 1]
        processed_images = (generated_images.detach() + 1.0) / 2.0
        processed_images = torch.clamp(processed_images, 0.0, 1.0)
        
        # 2. Resize to ear model's expected input size (e.g., 112x112)
        # TF.resize works on tensors (B, C, H, W)
        resized_images_for_ear_model = TF.resize(processed_images, EAR_MODEL_INPUT_SIZE, antialias=True)
        
        # 3. Normalize for ear model (e.g., from [0,1] to [-1,1] if it's ArcFace-like)
        # The ear_recognition_transform_to_tensor_and_norm expects [0,1] input if ToTensor was part of its PIL pipeline
        normalized_images_for_ear_model = ear_recognition_transform_to_tensor_and_norm(resized_images_for_ear_model)
        
        # 4. Get embeddings using the ear recognition model
        with torch.no_grad(): # Ensure no gradients for the ear model if it's also on CUDA
             generated_embeddings_raw = egd_model_instance.ear_recognition_model(normalized_images_for_ear_model.to(egd_model_instance.device))
        generated_embeddings = F.normalize(generated_embeddings_raw, p=2, dim=1) # L2 normalize

    # Ensure target_embeddings are also L2 normalized (should be if they come from ArcFace)
    target_embeddings_normalized = F.normalize(target_embeddings.to(generated_embeddings.device), p=2, dim=1)
    
    # Calculate cosine similarity loss
    similarity = F.cosine_similarity(generated_embeddings, target_embeddings_normalized, dim=1)
    
    loss = weight * (1.0 - similarity.mean())
    return loss

def main_train(config):
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    run_name = f"lr{config.LEARNING_RATE}_bs{config.TRAIN_BATCH_SIZE}_epochs{config.NUM_TRAIN_EPOCHS}_emb{config.EMBEDDING_DIM}_idloss{config.USE_IDENTITY_CONSISTENCY}"
    output_dir = os.path.join(config.OUTPUT_DIR_BASE, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load your pre-trained ear recognition model ---
    ear_recognition_model_instance = None
    if config.USE_IDENTITY_CONSISTENCY:
        try:
            # Ensure config has EAR_MODEL_PATH, EAR_MODEL_NAME, EAR_EMBEDDING_SIZE
            print(f"Loading ear recognition model from: {config.EAR_MODEL_PATH}")
            ear_recognition_model_instance = get_ear_recognition_model(
                config.EAR_MODEL_NAME, 
                fp16=False,  # Typically False for inference unless trained with it and supported
                num_features=config.EAR_EMBEDDING_SIZE
            )
            # checkpoint = torch.load(config.EAR_MODEL_PATH, map_location=device)
            # # Handle potential 'module.' prefix if saved from DataParallel/DistributedDataParallel
            # state_dict = checkpoint.get('state_dict', checkpoint) # Common key, or might be checkpoint itself
            # cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # ear_recognition_model_instance.load_state_dict(cleaned_state_dict)
            ear_recognition_model_instance.load_state_dict(torch.load(config.EAR_MODEL_PATH, map_location=device)) # Simpler load if not DDP
            
            ear_recognition_model_instance = ear_recognition_model_instance.to(device)
            ear_recognition_model_instance.eval()
            print("Ear recognition model loaded successfully for identity loss.")
        except Exception as e:
            print(f"Error loading ear recognition model: {e}. Identity consistency loss will be disabled or use placeholder.")
            ear_recognition_model_instance = None
            # config.USE_IDENTITY_CONSISTENCY = False # Optionally disable if model load fails

    egd_model = EmbeddingGuidedDiffusion(
        model_id=config.MODEL_ID,
        local_dir_base=config.LOCAL_MODEL_CACHE_DIR,
        embedding_dim=config.EMBEDDING_DIM,
        helper_prompt="",
        device=device,
        torch_dtype=torch.float16 if config.MIXED_PRECISION else torch.float32,
        ).to(device)
    
    if config.MIXED_PRECISION:
        # Keep model in FP16 for memory efficiency, but ensure parameters have FP32 gradients
        for param in egd_model.unet.parameters():
            param.data = param.data.to(torch.float32)
        
    # Assign the loaded ear recognition model to the EGD model instance
    if ear_recognition_model_instance:
        egd_model.ear_recognition_model = ear_recognition_model_instance


    dataset = EarGenDataset(
        feature_vectors_path=config.FEATURE_VECTORS_PATH, # These should be from your ear model
        image_dir=config.IMAGE_DIR,
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        image_file_extension=config.IMAGE_FILE_EXTENSION
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_DATALOADER_WORKERS 
    )

    optimizer = torch.optim.AdamW(
        egd_model.unet.parameters(), 
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        weight_decay=config.ADAM_WEIGHT_DECAY,
        eps=config.ADAM_EPSILON,
    )

    # VAE and noise_scheduler setup
    vae = egd_model.vae
    # For training, VAE might be better in float32 for stability during encode/decode if mixed precision is an issue
    # vae = vae.to(torch.float32) # Or keep as egd_model.torch_dtype if stable
    noise_scheduler = egd_model.scheduler
    
    # Ensure VAE and text_encoder grads are off
    vae.requires_grad_(False)
    egd_model.text_encoder.requires_grad_(False)

    print("Starting fine-tuning...")
    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=config.MIXED_PRECISION)


    for epoch in range(config.NUM_TRAIN_EPOCHS):
        egd_model.unet.train() 
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{config.NUM_TRAIN_EPOCHS}")

        for step, batch in enumerate(dataloader):
            feature_vectors = batch['feature_vector'].to(device, dtype=torch.float32) # From dataset
            target_images_tensor = batch['image_tensor'].to(device) # From dataset, typically [-1,1]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=config.MIXED_PRECISION):
                with torch.no_grad():
                    # VAE is usually float32 or float16. Ensure input matches VAE dtype.
                    latents = vae.encode(target_images_tensor.to(dtype=vae.dtype)).latent_dist.sample() 
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # feature_vectors are used to create prompt_embeds inside egd_model
                noise_pred = egd_model(noisy_latents, timesteps, feature_vectors_batch=feature_vectors) 
                
                mse_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                total_loss = mse_loss
                id_loss_value = 0.0
                
                if config.USE_IDENTITY_CONSISTENCY and egd_model.ear_recognition_model:
                    # Different versions of diffusers have different scheduler APIs
                    # We need to manually calculate the predicted denoised latents
                    
                    # 1. Get alpha_prod_t and beta_prod_t for the current timesteps
                    alpha_prod_t = noise_scheduler.alphas_cumprod.gather(0, timesteps)
                    beta_prod_t = 1 - alpha_prod_t
                    
                    # 2. Ensure proper dimensions using unsqueeze instead of reshape
                    # This is more explicit about which dimensions we're adding
                    alpha_prod_t = alpha_prod_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    beta_prod_t = beta_prod_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    
                    # 3. Predict x_0 from x_t using the noise prediction
                    # For debugging, print the shapes of tensors
                    # print(f"Shapes - noisy_latents: {noisy_latents.shape}, noise_pred: {noise_pred.shape}, alpha: {alpha_prod_t.shape}")
                    
                    # This replaces noise_scheduler.predict_start_from_noise()
                    pred_denoised_latents = (noisy_latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
                    
                    with torch.no_grad():
                        # Decode to images, ensure dtype compatibility with VAE
                        generated_images_for_loss = vae.decode((pred_denoised_latents / vae.config.scaling_factor).to(dtype=vae.dtype)).sample.float()
                    
                    id_loss = identity_consistency_loss(
                        egd_model, 
                        generated_images_for_loss, 
                        feature_vectors, # These are the target embeddings
                        weight=config.IDENTITY_LOSS_WEIGHT
                    )
                    total_loss = total_loss + id_loss
                    id_loss_value = id_loss.item()

            scaler.scale(total_loss).backward()
            
            if config.MAX_GRAD_NORM > 0:
                if config.MIXED_PRECISION:
                    # Skip unscaling for FP16 gradients - just skip gradient clipping when using mixed precision
                    torch.nn.utils.clip_grad_norm_(egd_model.unet.parameters(), config.MAX_GRAD_NORM * scaler.get_scale())
                else:
                    # Only unscale and clip if we're not using mixed precision
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(egd_model.unet.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            progress_bar.update(1)
            
            log_dict = {"mse_loss": mse_loss.item()}
            if config.USE_IDENTITY_CONSISTENCY and egd_model.ear_recognition_model:
                log_dict["id_loss"] = id_loss_value
            log_dict["total_loss"] = total_loss.item()
            progress_bar.set_postfix(log_dict)

        progress_bar.close()
        if (epoch + 1) % config.SAVE_EPOCH_FREQ == 0: # Use a config param for save frequency
            epoch_save_path = os.path.join(output_dir, f"unet_epoch_{epoch+1}")
            egd_model.unet.save_pretrained(epoch_save_path)
            print(f"Saved UNet checkpoint for epoch {epoch+1} to {epoch_save_path}")

    print("Fine-tuning finished.")
    final_save_path = os.path.join(output_dir, "unet_final")
    egd_model.unet.save_pretrained(final_save_path)
    print(f"Saved final UNet model to {final_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune EmbeddingGuidedDiffusion model for ear generation.")
    
    # General training parameters
    parser.add_argument("--LEARNING_RATE", type=float, default=p.LEARNING_RATE)
    parser.add_argument("--NUM_TRAIN_EPOCHS", type=int, default=p.NUM_TRAIN_EPOCHS)
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=p.TRAIN_BATCH_SIZE)
    parser.add_argument("--OUTPUT_DIR_BASE", type=str, default=p.OUTPUT_DIR_BASE)
    parser.add_argument("--SEED", type=int, default=p.SEED)
    parser.add_argument("--IMAGE_SIZE", type=int, default=p.IMAGE_SIZE)
    parser.add_argument("--MIXED_PRECISION", type=bool, default=getattr(p, "MIXED_PRECISION", True)) # Default to True if not in params
    parser.add_argument("--GRADIENT_ACCUMULATION_STEPS", type=int, default=getattr(p, "GRADIENT_ACCUMULATION_STEPS", 1))
    parser.add_argument("--MAX_GRAD_NORM", type=float, default=getattr(p, "MAX_GRAD_NORM", 1.0))
    parser.add_argument("--SAVE_EPOCH_FREQ", type=int, default=getattr(p, "SAVE_EPOCH_FREQ", 10))


    # Model and data parameters
    parser.add_argument("--MODEL_ID", type=str, default=p.MODEL_ID)
    parser.add_argument("--LOCAL_MODEL_CACHE_DIR", type=str, default=p.LOCAL_MODEL_CACHE_DIR)
    parser.add_argument("--EMBEDDING_DIM", type=int, default=p.EMBEDDING_DIM, help="Dimension of ear embeddings from your model")
    parser.add_argument("--FEATURE_VECTORS_PATH", type=str, default=p.FEATURE_VECTORS_PATH)
    parser.add_argument("--IMAGE_DIR", type=str, default=p.IMAGE_DIR)
    parser.add_argument("--IMAGE_FILE_EXTENSION", type=str, default=p.IMAGE_FILE_EXTENSION)
    parser.add_argument("--NUM_DATALOADER_WORKERS", type=int, default=p.NUM_DATALOADER_WORKERS)


    # AdamW optimizer parameters
    parser.add_argument("--ADAM_BETA1", type=float, default=p.ADAM_BETA1)
    parser.add_argument("--ADAM_BETA2", type=float, default=p.ADAM_BETA2)
    parser.add_argument("--ADAM_WEIGHT_DECAY", type=float, default=p.ADAM_WEIGHT_DECAY)
    parser.add_argument("--ADAM_EPSILON", type=float, default=p.ADAM_EPSILON)

    # Identity consistency loss parameters
    parser.add_argument("--USE_IDENTITY_CONSISTENCY", type=bool, default=getattr(p, "USE_IDENTITY_CONSISTENCY", False))
    parser.add_argument("--IDENTITY_LOSS_WEIGHT", type=float, default=getattr(p, "IDENTITY_LOSS_WEIGHT", 1.0))
    parser.add_argument("--EAR_MODEL_PATH", type=str, default=getattr(p, "EAR_MODEL_PATH", '/home/aljazjustin/ear_generation/models_embeding/vit16_corrected_margin_epoh7/model.pt'))
    parser.add_argument("--EAR_MODEL_NAME", type=str, default=getattr(p, "EAR_MODEL_NAME", 'vit_b')) # e.g., 'vit_b', 'r50'
    parser.add_argument("--EAR_EMBEDDING_SIZE", type=int, default=getattr(p, "EAR_EMBEDDING_SIZE", 1024)) # Output dim of your ear model

    # Create a namespace from params.py defaults
    config_ns = argparse.Namespace()
    for param_name, default_value in p.__dict__.items():
        if not param_name.startswith('_'): # Exclude private/magic attributes
            setattr(config_ns, param_name, default_value)
    
    # Parse known args to update the namespace, allowing CLI overrides
    args = parser.parse_args()
    for k, v in vars(args).items():
        if hasattr(config_ns, k): # Only update if it's a known param from params.py or parser
             setattr(config_ns, k, v)
        else: # If argparser has a new param not in params.py, add it
             setattr(config_ns, k, v)
            
    main_train(config_ns)
