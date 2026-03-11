"""
Global configurations for CyIN (Cyclic Informative Latent Space)
Based on the framework from original_ITHP
"""
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Modality dimensions (will be set by set_dataset_config)
TEXT_DIM = 0
ACOUSTIC_DIM = 0
VISUAL_DIM = 0

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Default hyperparameters for CyIN (Table 6 - MOSI)
DEFAULT_CONFIG = {
    # Dimension C_U for Unimodal Representation
    'unified_dim': 256,
    # Dimension C_ib in IB Encoder or Decoder
    'ib_dim': 256,
    # Dimension C_B of Bottleneck Latent
    'bottleneck_dim': 128,
    # IB trade-off coefficient (β) - MOSI: 32
    'beta': 32,
    # Translation loss coefficient (γ) - MOSI: 10
    'gamma': 10,
    # Num Layers of RA() in CRA - MOSI: 8
    'cra_layers': 8,
    # Dimension of RA() in CRA - MOSI: [64, 32, 16]
    'cra_dims': [64, 32, 16],
    # Num Layers of Cross-modal Attention - MOSI: 2
    'attention_layers': 2,
    # Num Heads of Attention H - MOSI: 2
    'attention_heads': 2,
    # Dropout probability
    'dropout_prob': 0.3,
    # Training stage ratio 1st:2nd - MOSI: 1:9
    'stage1_ratio': 0.1,
}


def set_dataset_config(dataset_name):
    """Set dataset-specific configurations"""
    global TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM

    dataset_configs = {
        "mosi": {"ACOUSTIC_DIM": 74, "VISUAL_DIM": 47, "TEXT_DIM": 768},
        "mosi_imagebind": {"ACOUSTIC_DIM": 1024, "VISUAL_DIM": 1024, "TEXT_DIM": 768},
        "mosei": {"ACOUSTIC_DIM": 74, "VISUAL_DIM": 35, "TEXT_DIM": 768},
        "iemocap": {"ACOUSTIC_DIM": 74, "VISUAL_DIM": 35, "TEXT_DIM": 768},
        "meld": {"ACOUSTIC_DIM": 74, "VISUAL_DIM": 35, "TEXT_DIM": 768},
    }

    config = dataset_configs.get(dataset_name.lower())
    if config:
        ACOUSTIC_DIM = config["ACOUSTIC_DIM"]
        VISUAL_DIM = config["VISUAL_DIM"]
        TEXT_DIM = config["TEXT_DIM"]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
