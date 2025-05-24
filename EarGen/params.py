# --- Paths ---

IMAGE_DIR = "path/to/ears"
OUTPUT_DIR_BASE = "path/to/models_finetuned" # Base, can be appended with run-specific names
LOCAL_MODEL_CACHE_DIR = "path/to/models"

EMBEDDING_DIM = 768  # MUST match the output dimension of your ear recognition model
MODEL_ID = "runwayml/stable-diffusion-v1-5" # Or your base model
LOCAL_MODEL_CACHE_DIR = "path/to/models_diffusers_cache" # Example path

# Training Hyperparameters
LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 2 # Adjust based on your GPU memory
MIXED_PRECISION = True # Use mixed precision training
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0 # Set to 0 to disable
SAVE_EPOCH_FREQ = 5 # How often to save checkpoints

# Dataset paths
FEATURE_VECTORS_PATH = "path/to/feature_vectors_arc.json" # Path to your .json embeddings
IMAGE_DIR = "path/to/ears-by-fols" 
IMAGE_FILE_EXTENSION = ".png" # Or .jpg, etc.
IMAGE_SIZE = 512 # Size of images for diffusion model training
NUM_DATALOADER_WORKERS = 4

# AdamW Optimizer
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_WEIGHT_DECAY = 1e-2
ADAM_EPSILON = 1e-08

# Identity Consistency Loss (Optional)
USE_IDENTITY_CONSISTENCY = True # Set to True to enable
IDENTITY_LOSS_WEIGHT = 0.25 # Weight for the identity loss component
EAR_MODEL_PATH = 'path/to/model.pt' # Path to your trained ear model
EAR_MODEL_NAME = 'vit_b'  # Name of the ear model architecture (e.g., 'r50', 'vit_b' for get_model)
EAR_EMBEDDING_SIZE = 768 # Output dimension of your ear model (should be same as EMBEDDING_DIM)


RESUME_FROM_CHECKPOINT = True
CHECKPOINT_PATH = "path/to"  # Path to your best checkpoint


# Other params
SEED = 42
OUTPUT_DIR_BASE = "path/to/eargen_arc2face"

# These are not directly used by the new prepare_prompt_embeds method but might be in old code parts
HELPER_PROMPT = "" # Not used with placeholder token method
NUM_FEATURE_TOKENS = 1 # Implicitly 1 for the placeholder token
REPEAT_FEATURE_VECTOR = 1 # Not used