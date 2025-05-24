import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import logging as diffusers_logging
from transformers import CLIPTextModel, CLIPTokenizer
import os

# Import parameters for default values if desired
import params as p # Assuming params.py is in the same directory

# Optional: Suppress verbose diffusers logging if desired
# diffusers_logging.set_verbosity_error()

class CLIPTextModelWrapper(nn.Module):
    """Wrapper for CLIP text encoder that enables embedding injection"""
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder_actual = text_encoder
        self.config = text_encoder.config
    
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        
        # If inputs_embeds is provided, we need to access the text model components directly
        if inputs_embeds is not None:
            # Access CLIP text model components directly
            if hasattr(self.text_encoder_actual, 'text_model'):
                text_model = self.text_encoder_actual.text_model
                
                # Forward through encoder component directly
                if hasattr(text_model, 'encoder'):
                    encoder_outputs = text_model.encoder(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=True
                    )
                    hidden_states = encoder_outputs[0]
                    
                    # Apply final layer norm if present
                    if hasattr(text_model, 'final_layer_norm'):
                        hidden_states = text_model.final_layer_norm(hidden_states)
                    
                    # Return in the same format as CLIPTextModel
                    from transformers.modeling_outputs import BaseModelOutputWithPooling
                    return BaseModelOutputWithPooling(
                        last_hidden_state=hidden_states,
                        pooler_output=hidden_states[:, 0],  # First token as pooler output
                        hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                        attentions=encoder_outputs.attentions if output_attentions else None
                    )
                else:
                    raise ValueError("Text model structure not as expected. Missing encoder component.")
            else:
                raise ValueError("Text encoder structure not as expected. Missing text_model attribute.")
        elif input_ids is not None:
            # Normal operation if input_ids are provided
            return self.text_encoder_actual(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

    def get_input_embeddings(self):
        return self.text_encoder_actual.get_input_embeddings()



class EmbeddingGuidedDiffusion(nn.Module):
    def __init__(self,
                 model_id=p.MODEL_ID,
                 local_dir_base=p.LOCAL_MODEL_CACHE_DIR,
                 embedding_dim=p.EMBEDDING_DIM, 
                 helper_prompt=p.HELPER_PROMPT, 
                 device=None,
                 torch_dtype=torch.float16,
                 num_feature_tokens=p.NUM_FEATURE_TOKENS, 
                 repeat_feature_vector=p.REPEAT_FEATURE_VECTOR 
                ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype 
        
        self.embedding_dim = embedding_dim 

        model_name = model_id.split('/')[-1]
        local_dir = os.path.join(local_dir_base, model_name)
        os.makedirs(local_dir, exist_ok=True)

        # Load the base text_encoder first
        base_text_encoder = CLIPTextModel.from_pretrained(
            model_id if not os.path.exists(os.path.join(local_dir, "text_encoder")) else os.path.join(local_dir, "text_encoder"),
            subfolder="text_encoder" if not os.path.exists(os.path.join(local_dir, "text_encoder")) else "",
        ).to(self.device, dtype=self.torch_dtype)

        # Wrap the text_encoder
        self.text_encoder = CLIPTextModelWrapper(base_text_encoder)


        if os.path.exists(os.path.join(local_dir, "unet", "config.json")):
            print(f"Loading model components from local cache: {local_dir}")
            self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(local_dir, "tokenizer"))
            # self.text_encoder is already initialized and wrapped above
            self.vae = AutoencoderKL.from_pretrained(os.path.join(local_dir, "vae")).to(self.device, dtype=self.torch_dtype)
            self.unet = UNet2DConditionModel.from_pretrained(os.path.join(local_dir, "unet")).to(self.device, dtype=self.torch_dtype)
        else:
            print(f"Downloading model components from HuggingFace: {model_id}")
            self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            # self.text_encoder is already initialized and wrapped above
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device, dtype=self.torch_dtype)
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device, dtype=self.torch_dtype)
            
            print(f"Saving model components to local cache: {local_dir}")
            self.tokenizer.save_pretrained(os.path.join(local_dir, "tokenizer"))
            # Save the underlying text_encoder if you want to save the original
            self.text_encoder.text_encoder_actual.save_pretrained(os.path.join(local_dir, "text_encoder")) 
            self.vae.save_pretrained(os.path.join(local_dir, "vae"))
            self.unet.save_pretrained(os.path.join(local_dir, "unet"))

        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        self.placeholder_token_str = "<ear_identity_token>"
        num_added_tokens = self.tokenizer.add_tokens(self.placeholder_token_str)
        if num_added_tokens > 0:
            print(f"Added placeholder token: {self.placeholder_token_str}")
            # Resize embeddings of the *actual* underlying text encoder
            self.text_encoder.text_encoder_actual.resize_token_embeddings(len(self.tokenizer))
        self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(self.placeholder_token_str)

        self.text_encoder_dim = self.text_encoder.config.hidden_size 
        self.max_seq_len = self.tokenizer.model_max_length 

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False) # The wrapper and its underlying model
        
        self.unet.train() 
        print("EmbeddingGuidedDiffusion model initialized. UNet is in training mode.")
        self.ear_recognition_model = None

    def prepare_prompt_embeds(self, feature_vectors_batch):
        batch_size = feature_vectors_batch.shape[0]
        device = self.device
        dtype = self.torch_dtype

        ear_embeddings = feature_vectors_batch.to(device, dtype=dtype)

        prompt_text = f"a photo of an {self.placeholder_token_str} ." 
        
        text_inputs = self.tokenizer(
            [prompt_text] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        # First get standard embeddings from the text encoder
        with torch.no_grad():
            encoder_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            base_embeddings = encoder_outputs.last_hidden_state.clone()
        
        # Process ear embeddings to match dimensions
        text_encoder_hidden_size = self.text_encoder.config.hidden_size
        if ear_embeddings.shape[-1] != text_encoder_hidden_size:
            padding_size = text_encoder_hidden_size - ear_embeddings.shape[-1]
            if padding_size > 0:
                ear_embeddings_processed = torch.nn.functional.pad(
                    ear_embeddings, (0, padding_size), "constant", 0
                )
            else: 
                ear_embeddings_processed = ear_embeddings[:, :text_encoder_hidden_size]
        else:
            ear_embeddings_processed = ear_embeddings
        
        if ear_embeddings_processed.ndim == 1: 
            ear_embeddings_processed = ear_embeddings_processed.unsqueeze(0)

        # Find placeholder token positions and replace with ear embeddings
        for i in range(batch_size):
            placeholder_indices = (input_ids[i] == self.placeholder_token_id).nonzero(as_tuple=True)[0]
            if placeholder_indices.numel() > 0:
                idx_to_replace = placeholder_indices[0] 
                # Replace the embedding at the placeholder position
                base_embeddings[i, idx_to_replace, :] = ear_embeddings_processed[i, :]
            else:
                print(f"Warning: Placeholder token ID {self.placeholder_token_id} ({self.placeholder_token_str}) not found in input_ids for batch item {i}.")

        return base_embeddings.to(dtype)



    def forward(self, noisy_latents, timesteps, feature_vectors_batch=None, prompt_embeds=None):
        if prompt_embeds is None:
            if feature_vectors_batch is None:
                raise ValueError("Either feature_vectors_batch or prompt_embeds must be provided.")
            prompt_embeds = self.prepare_prompt_embeds(feature_vectors_batch)

        noisy_latents_casted = noisy_latents.to(self.torch_dtype)
        prompt_embeds_casted = prompt_embeds.to(self.torch_dtype)
        
        noise_pred = self.unet(
            noisy_latents_casted, 
            timesteps,
            encoder_hidden_states=prompt_embeds_casted
        ).sample
        return noise_pred

    @torch.no_grad()
    def generate_image(self, 
                       feature_vectors_batch=None, 
                       prompt_embeds_input=None, 
                       num_inference_steps=50, 
                       guidance_scale=7.5,
                       height=p.IMAGE_SIZE, 
                       width=p.IMAGE_SIZE,
                       generator=None): 
        if prompt_embeds_input is None:
            if feature_vectors_batch is None:
                raise ValueError("Either feature_vectors_batch or prompt_embeds_input must be provided for generation.")
            if feature_vectors_batch.ndim == 1: 
                feature_vectors_batch = feature_vectors_batch.unsqueeze(0)
            prompt_embeds_input = self.prepare_prompt_embeds(feature_vectors_batch)
        
        batch_size = prompt_embeds_input.shape[0]

        if guidance_scale > 1.0:
            uncond_prompt_text = "" 
            max_length = prompt_embeds_input.shape[1]
            
            uncond_input_ids = self.tokenizer(
                [uncond_prompt_text] * batch_size,
                padding="max_length",
                max_length=max_length, 
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            
            # Get standard unconditional embeddings using the wrapper
            # The wrapper's forward should handle input_ids correctly
            uncond_outputs = self.text_encoder(input_ids=uncond_input_ids, return_dict=True)
            uncond_embeddings = uncond_outputs.last_hidden_state.to(self.torch_dtype)
            
            combined_embeds = torch.cat([uncond_embeddings, prompt_embeds_input])
        else:
            combined_embeds = prompt_embeds_input

        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=self.torch_dtype # Use self.torch_dtype
        )
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            # Ensure latent_model_input is also casted if necessary, though scheduler.scale_model_input might handle it
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).to(self.torch_dtype)


            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=combined_embeds.to(self.torch_dtype)).sample # ensure combined_embeds is correct dtype

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        latents = 1 / self.vae.config.scaling_factor * latents
        # Ensure VAE output is float before clamping, decode usually outputs float
        images = self.vae.decode(latents.to(self.vae.dtype)).sample.float() # cast latents to VAE dtype
        
        images = (images / 2 + 0.5).clamp(0, 1) 
        return images