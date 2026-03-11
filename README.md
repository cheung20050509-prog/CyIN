# CyIN: Cyclic Informative Latent Space

Reproduction of **CyIN** (NeurIPS 2024) with a DeBERTa-V3 backbone for multimodal sentiment analysis.

## Key Components

- `CyIN.py` — Core framework: Information Bottleneck (token-level & label-level), Cascaded Residual Autoencoder (CRA), Cross-Modal Attention fusion
- `deberta_CyIN.py` — DeBERTa-V3 integration wrapper
- `train.py` — Two-stage training pipeline (Stage 1: IB space construction, Stage 2: cross-modal translation)
- `test.py` — Evaluation script
- `global_configs.py` — Dataset-specific configurations

## Usage

### Training (MOSI)

```bash
bash run.sh
```

### Testing

```bash
bash test.sh
```

### ImageBind Features

```bash
bash run_imagebind.sh
bash test_imagebind.sh
```

## Results on MOSI (Complete Modality)

| Model | Acc2 | F1 | MAE | Corr |
|-------|------|----|-----|------|
| CyIN (Paper, BERT) | 86.3 | 86.3 | 0.712 | 0.801 |
| CyIN (Ours, DeBERTa-V3) | 86.7 | 86.6 | 0.791 | 0.805 |

## Requirements

```bash
pip install -r requirements.txt
```

## Reference

> CyIN: Bridging Complete and Incomplete Multimodal Learning via Cyclic Informative Latent Space (NeurIPS 2024)
