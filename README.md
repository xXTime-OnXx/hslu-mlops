# HSLU MLOPS - Project 2

This repository contains a containerized machine learning pipeline for fine-tuning DistilBERT on paraphrase detection using the MRPC dataset.

## Installation

### Local
1. Clone this repository:
```bash
git clone https://github.com/xXTime-OnXx/hslu-mlops.git
cd hslu-mlops
```

2. (Optional) Create a virtual environment and activate it

3. Install required python packages
```bash
pip install -r requirements.txt
```

### Usage
```bash
python train.py [OPTIONS]
```

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--learning_rate` | float | `1e-4` | Learning rate for optimizer |
| `--adam_beta1` | float | `0.85` | Beta1 parameter for Adam/AdamW optimizer |
| `--adam_beta2` | float | `0.999` | Beta2 parameter for Adam/AdamW optimizer |
| `--adam_epsilon` | float | `1e-8` | Epsilon parameter for Adam/AdamW optimizer |
| `--warmup_steps` | int | `10` | Number of warmup steps for learning rate scheduler |
| `--lr_schedule` | str | `linear` | Learning rate schedule (`linear`, `cosine`, `onecycle`, `constant`) |
| `--weight_decay` | float | `0.0` | Weight decay for optimizer |
| `--optimizer` | str | `adamw` | Optimizer type (`adam`, `adamw`) |

### Example
```bash
python train.py --learning_rate 1e-3 --adam_beta1 0.9 --warmup_steps 15
```

### Viewing Training Results

Training metrics are logged to the experiment tracking tool (wandb). 
Check the console output for the tracking URL where you can monitor training progress and results.