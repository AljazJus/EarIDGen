# EarIDGen
Generation of ear identities using diffusion model

## Overview

EarIDGen is a research project that explores the effectiveness of generating synthetic ear images from feature embeddings for biometric identification purposes. This work combines diffusion models with embedding-guided generation to create identity-preserving ear images that maintain biometric characteristics suitable for recognition tasks.

## Abstract

This project trains a generative model to reconstruct ear images from extracted feature vectors and evaluates its ability to preserve identity-specific characteristics. Two evaluation strategies are employed:
1. Cosine similarity-based embedding model evaluation
2. Siamese network-based comparison evaluation

The research demonstrates that generated images maintain high similarity with their corresponding real images while preserving distinct inter-class separability, showing potential for embedding-based generation methods in biometric systems.

## Project Structure

```
EarIDGen/
├── data_preparation/           # Data preprocessing and feature extraction
│   └── img2vec_arc.py         # ArcFace-based feature vector extraction
├── EarGen/                    # Core generation and evaluation modules
│   ├── EarGenDataset.py       # Dataset handling for training
│   ├── EGD.py                 # Embedding Guided Diffusion model
│   ├── fine_tune.py           # Model fine-tuning script
│   ├── fine_tune_checkpoint.py # Checkpoint-based fine-tuning
│   ├── params.py              # Configuration parameters
│   ├── siamese.py             # Siamese network for similarity evaluation
│   ├── test_gen.py            # Basic generation testing
│   ├── test_gen_comparision.py # Embedding-based evaluation
│   └── test_gen_comparision_siamese.py # Siamese-based evaluation
├── embeding_model/            # Embedding model configurations
│   └── ArcVec_config.py       # ArcFace model configuration
└── README.md
```

## Key Components

### 1. Feature Extraction ([`data_preparation/img2vec_arc.py`](data_preparation/img2vec_arc.py))
- Extracts feature vectors from ear images using ArcFace models
- Supports ViT-based architectures for robust feature representation
- Outputs features in JSON format for training and evaluation

### 2. Embedding Guided Diffusion ([`EarGen/EGD.py`](EarGen/EGD.py))
- Core generative model that creates ear images from feature embeddings
- Built on Stable Diffusion architecture with custom embedding integration
- Supports identity consistency loss for improved generation quality

### 3. Evaluation Framework
- **Embedding-based evaluation** ([`test_gen_comparision.py`](EarGen/test_gen_comparision.py)): Uses cosine similarity on feature embeddings
- **Siamese-based evaluation** ([`test_gen_comparision_siamese.py`](EarGen/test_gen_comparision_siamese.py)): Uses trained Siamese networks for similarity assessment

### 4. Configuration Management ([`EarGen/params.py`](EarGen/params.py))
- Centralized parameter configuration for all training and evaluation settings
- Supports various hyperparameters for diffusion training and model architecture

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EarIDGen.git
cd EarIDGen
```

2. Install dependencies:
```bash
pip install torch torchvision diffusers transformers
pip install PIL numpy matplotlib pandas
pip install safetensors easydict
```

3. Install ArcFace dependencies:
```bash
# Add ArcFace implementation to your Python path
# Update sys.path.insert() calls in the code with your actual paths
```

## Usage

### 1. Feature Extraction

First, extract features from your ear image dataset:

```bash
cd data_preparation
python img2vec_arc.py
```

This will generate a `feature_vectors_arc.json` file containing embeddings for all images.

### 2. Model Training

Configure your training parameters in [`EarGen/params.py`](EarGen/params.py), then run:

```bash
cd EarGen
python fine_tune.py
```

For checkpoint-based training:
```bash
python fine_tune_checkpoint.py
```

### 3. Evaluation

#### Embedding-based Evaluation:
```bash
python test_gen_comparision.py --run_name your_model_run --epoch unet_epoch_50
```

#### Siamese-based Evaluation:
```bash
python test_gen_comparision_siamese.py \
  --run_name your_model_run \
  --epoch unet_epoch_50 \
  --siamese_model_path path/to/siamese_model.pth \
  --num_classes 10
```

## Configuration

### Key Parameters ([`EarGen/params.py`](EarGen/params.py))

```python
# Model Configuration
EMBEDDING_DIM = 768  # Must match your embedding model output
MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMAGE_SIZE = 512

# Training Parameters
LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 2
USE_IDENTITY_CONSISTENCY = True
IDENTITY_LOSS_WEIGHT = 0.25

# Data Paths
FEATURE_VECTORS_PATH = "path/to/feature_vectors_arc.json"
IMAGE_DIR = "path/to/ears-by-fols"
```

### ArcFace Model Configuration ([`embeding_model/ArcVec_config.py`](embeding_model/ArcVec_config.py))

```python
config.embedding_size = 768
config.network = "vit_b"
config.margin_list = (1.0, 0.0, 0.4)  # ArcFace margin settings
config.num_classes = 660  # Adjust based on your dataset
```

## Evaluation Metrics

The framework provides comprehensive evaluation through:

1. **Intra-class Analysis**: Similarity between generated and real images of the same identity
2. **Inter-class Analysis**: Discriminative power between different identities
3. **Statistical Analysis**: Mean, standard deviation, and distribution analysis

### Output Metrics:
- Intra-class Generated vs Original similarity
- Intra-class Generated vs Generated similarity
- Inter-class Generated vs Original similarity
- Inter-class similarity distributions

## Research Applications

This framework supports research in:
- Biometric data augmentation
- Privacy-preserving biometric systems
- Identity-preserving image generation
- Synthetic biometric evaluation

## Dependencies

- PyTorch >= 1.9.0
- Diffusers
- Transformers
- PIL (Pillow)
- NumPy
- Matplotlib
- Pandas
- SafeTensors
- EasyDict

## License

This project is released under [appropriate license]. Please cite our work if you use this code in your research.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{earidgen2024,
  title={Generation of Ear Identities using Diffusion Models: An Embedding-Based Approach},
  author={[Your Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Contact

For questions or collaboration opportunities, please contact [your contact information].