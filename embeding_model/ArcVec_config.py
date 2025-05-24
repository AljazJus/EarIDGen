from easydict import EasyDict as edict

config = edict()
config.dataset = "ear_dataset"
config.embedding_size = 768 # Or 768, more standard for ViT-B. 1024 is also possible if intended.
config.sample_rate = 1.0
config.fp16 = False
config.optimizer = "adamw"
config.momentum = 0.9 # Momentum is not strictly used by AdamW in the same way as SGD, but often kept in configs.
config.weight_decay = 0.05 # ViTs often benefit from stronger weight decay with AdamW
config.batch_size = 256  # Adjust based on your GPU memory

# Learning Rate: Consider starting a bit lower for ViT with AdamW, or keep 0.001 and monitor
config.lr = 0.0005 # Example: 5e-4 or 3e-4 are common for ViT fine-tuning

config.frequent = 10  # Log training stats every 10 batches
config.verbose = 2000  # Run evaluation every N batches (if validation is set up)

config.rec = "/home/aljazjustin/ear_generation/data/SOURCES/VGGFace-Ear"
config.num_classes = 660
config.num_image = 234651
config.num_epoch = 20  # Increased number of epochs
config.warmup_epoch = 5   # Adjusted warmup epochs if total epochs change

config.scheduler = "cosine" 

config.output = "/home/aljazjustin/ear_generation/models_embeding/vit16_0.4" # New output directory

# --- CRITICAL CHANGE ---
# ArcFace margin: Use a standard value like 0.4 or 0.5.
# The other two values are for other margin types (CosFace, AM-Softmax) if CombinedMarginLoss supports them.
# If you only want ArcFace, they can often be 0.
config.margin_list = (1.0, 0.0, 0.4) # Example: ArcFace margin of 0.4 radians

config.network = "vit_b"
config.loss = "arcface" # This should ensure CombinedMarginLoss uses the ArcFace formulation with the first margin.

config.interclass_filtering_threshold = 0
config.num_workers = 4 # Increase if your system can handle it and data loading is a bottleneck
config.seed = 2024

# --- STRONGLY RECOMMENDED ---
# Add validation: Create a validation split of your ear data or use standard benchmarks if applicable.
# This requires preparing validation data in the format your dataloader expects (e.g., .rec/.idx or image folders)
# and listing them here. For example, if you have LFW (though it's for faces):
# config.val_targets = ['lfw'] 
# For custom ear validation, you'd need to implement the evaluation logic or adapt existing ones.
config.val_targets = [] # Keep empty if you don't have validation set up yet, but aim to add it.
