"""mlops_hyperparameter_tuning-timon_schmid.ipynb

# Hyperparameter Tuning
*(Note: This notebook runs significantly faster if you have access to a GPU. Use either the GPUHub, Google Colab, or your own GPU.)*

In this project, you will optimize the hyperparameters of a model in 3 stages.

## Paraphrase Detection
We finetune [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on [MRPC](https://huggingface.co/datasets/glue/viewer/mrpc/train), a paraphrase detection dataset. This notebook is adapted from a [PyTorch Lightning example](https://lightning.ai/docs/pytorch/1.9.5/notebooks/lightning_examples/text-transformers.html).
"""

import argparse

from datetime import datetime
from typing import Optional

import datasets
import evaluate
import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

import wandb
from lightning.pytorch.loggers import WandbLogger

from dotenv import load_dotenv
load_dotenv()

L.seed_everything(42)

MAX_EPOCHS = 3
BATCH_SIZE = 256  # Set to max that fits in GPU memory

RUN_SWEEP = False

wandb.login()

sweep_config = {
    'method': 'bayes',  # Changed from 'grid'
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        # Parameters to tune
        'learning_rate': {
            'min': 1e-5,    # Minimum value
            'max': 1e-3     # Maximum value
        },
        'adam_beta1': {
            'min': 0.75,
            'max': 0.90
        },
        'warmup_steps': {
            'min': 0,
            'max': 15
        },

        # Fixed parameters
        'lr_schedule': {
            'value': 'linear'
        },
        'weight_decay': {
            'value': 0.0
        },
        'optimizer': {
            'value': 'adamw'
        },
        'adam_epsilon': {
            'value': 1e-8
        },
        'adam_beta2': {
            'value': 0.999
        },
    }
}

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--adam_beta1', type=float, default=0.85)
parser.add_argument('--warmup_steps', type=int, default=10)
parser.add_argument('--lr_schedule', type=str, default='linear')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--optimizer', type=str, default='adamw')
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--adam_beta2', type=float, default=0.999)
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()

config = {
    "learning_rate": args.learning_rate,
    "adam_beta1": args.adam_beta1,
    "warmup_steps": args.warmup_steps,

    "lr_schedule": args.lr_schedule,
    "weight_decay": args.weight_decay,

    "optimizer": args.optimizer,
    "adam_epsilon": args.adam_epsilon,
    "adam_beta2": args.adam_beta2,
}

BATCH_SIZE = args.batch_size

# Define run name based on non-default script arguments
non_default_args = []
for action in parser._actions:
    if action.dest != 'help':
        arg_value = getattr(args, action.dest)
        if arg_value != action.default:
            non_default_args.append(f"{action.dest}_{arg_value}")

if non_default_args:
    run_name = f"distilbert-{'-'.join(non_default_args)}"
else:
    run_name = "distilbert-default"


class GLUEDataModule(L.LightningDataModule):
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding="max_length", truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]
        return features

class GLUETransformer(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        optimizer: str = "adamw",
        adam_beta1: float = 0.9,
        lr_schedule: str = "linear",
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})
        return val_loss

    def on_validation_epoch_end(self):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(self.validation_step_outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            self.validation_step_outputs.clear()
            return loss

        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # Choose optimizer with additional parameters
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.get('adam_epsilon', 1e-8),
                betas=(self.hparams.get('adam_beta1', 0.9), self.hparams.get('adam_beta2', 0.999))
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.get('adam_epsilon', 1e-8),
                betas=(self.hparams.get('adam_beta1', 0.9), self.hparams.get('adam_beta2', 0.999))
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        # Choose learning rate schedule
        if self.hparams.lr_schedule == "constant":
            return optimizer
        elif self.hparams.lr_schedule == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.hparams.lr_schedule == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.hparams.lr_schedule == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.get('max_lr', self.hparams.learning_rate * 10),
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.hparams.warmup_steps / self.trainer.estimated_stepping_batches if self.hparams.warmup_steps > 0 else 0.3,
            )
        else:
            raise ValueError(f"Unknown lr_schedule: {self.hparams.lr_schedule}")

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

def train():
    # Initialize wandb run (this gets config from sweep)
    wandb.init()

    # Get hyperparameters from wandb config
    config = wandb.config

    # Setup logger
    logger = WandbLogger(
        project="hslu-mlops-project1"
    )

    # Setup data module
    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )
    dm.setup("fit")

    # Create model with hyperparameters from sweep
    model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=config.learning_rate,
        lr_schedule=config.lr_schedule,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        adam_epsilon=config.adam_epsilon,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
    )

    # Train
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=logger,
    )
    trainer.fit(model, datamodule=dm)

    # Finish this run
    wandb.finish()

if RUN_SWEEP:
  sweep_id = wandb.sweep(
      sweep=sweep_config,
      project="hslu-mlops-project1",
      entity="timon-schmid-hochschule-luzern"
  )

  # Run the sweep agent
  # count: number of runs to execute (3x3x2 = 18 for your grid)
  wandb.agent(
      sweep_id,
      function=train,
      count=18  # Total number of runs (adjust based on your grid size)
  )

# Run single training with config
wandb.init(
    project="hslu-mlops-project1",
    name=run_name,
    config=config
)

# Setup data module
dm = GLUEDataModule(
    model_name_or_path="distilbert-base-uncased",
    task_name="mrpc",
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
)
dm.setup("fit")

# Create model with config
model = GLUETransformer(
    model_name_or_path="distilbert-base-uncased",
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    task_name=dm.task_name,
    learning_rate=config['learning_rate'],
    lr_schedule=config['lr_schedule'],
    warmup_steps=config['warmup_steps'],
    weight_decay=config['weight_decay'],
    optimizer=config['optimizer'],
    adam_epsilon=config['adam_epsilon'],
    adam_beta1=config['adam_beta1'],
    adam_beta2=config['adam_beta2'],
)

# Setup logger
logger = WandbLogger(
    project="hslu-mlops-project1",
    name=run_name,
)

# Train
trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=1,
    logger=logger,
)
trainer.fit(model, datamodule=dm)

# Finish the run
wandb.finish()