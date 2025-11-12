# HSLU MLOPS - Project 2

This repository contains a containerized machine learning pipeline for fine-tuning DistilBERT on paraphrase detection using the MRPC dataset.

## Available Parameters

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
| `--batch_size` | int | `256` | Batch size definition |

## Prerequisites
- WandB account (sign up at https://wandb.ai)
- Docker installed (for containerized execution)
- Python 3.11+ (for local execution)
- *(Optional) Virtual environment (for local execution)*

Clone this repository:
```bash
git clone https://github.com/xXTime-OnXx/hslu-mlops.git
cd hslu-mlops
```

## Local

### Installation
Install required python packages
```bash
pip install -r requirements.txt
```

Also make sure to set these environment variables
```bash
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=your_wandb_project
WANDB_ENTITY=your_wandb_entity
```

### Usage
```bash
python train.py [OPTIONS]
```

### Example
```bash
python train.py --learning_rate 1e-3 --adam_beta1 0.9 --warmup_steps 15
```

## Docker

### Installation
Run the following command to build the docker image:
```bash
docker build -t hslu-mlops-mrpc-training .
```

### Usage
```bash
# with default configuration
docker docker run -e WANDB_API_KEY=your_key -e WANDB_PROJECT=your_project -e WANDB_ENTITY=your_entity hslu-mlops-mrpc-training

# with custom configuration
docker run -e WANDB_API_KEY=your_key -e WANDB_PROJECT=your_project -e WANDB_ENTITY=your_entity hslu-mlops-mrpc-training python train.py [OPTIONS]
```

### Example
```bash
docker run -e WANDB_API_KEY=your_key -e WANDB_PROJECT=your_project -e WANDB_ENTITY=your_entity hslu-mlops-mrpc-training python train.py --learning_rate 1e-4 --adam_beta1 0.85 --warmup_steps 10 --lr_schedule linear --weight_decay 0.0 --optimizer adamw --batch_size 256
```

## Viewing Training Results

Training metrics are logged to the experiment tracking tool (wandb). 
Check the console output for the tracking URL where you can monitor training progress and results.