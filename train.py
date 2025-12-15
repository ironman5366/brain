# Builtin imports
import tomllib
from pathlib import Path
import sys
import typing

# Internal imports
from data.dataset import EEGDataset, SparseDataset
from models.mae import EEGViTMAE
from models.vit import EEGViT, EEGViTConfig
from models.mae import EEGMAEConfig, MAETrainer
from constants import DEFAULT_CHECKPOINT_DIR
from settings import WANDB_ENTITY, WANDB_PROJECT

# External imports
from pydantic import BaseModel
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup
from tqdm import tqdm


class Config(BaseModel):
    name: str
    data_path: str

    dataset: typing.Literal["standard"] | typing.Literal["sparse"] = "standard"

    # Model config
    mae: EEGMAEConfig
    vit: EEGViTConfig

    # Dataloading
    num_workers: int = 8
    batch_size: int = 1024
    shuffle: bool = True

    # Training
    epochs: int = 10
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)

    # Optimizer/scheduler hyperparams
    beta1: float = 0.9
    beta2: float = 0.999
    lr: float = 1e-3
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 32


def train(config: Config):
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name=WANDB_PROJECT,
        init_kwargs={"entity": WANDB_ENTITY},
        config=config.model_dump(),
    )

    print(f"Loading dataset/dataloader from {config.data_path}...")

    if config.dataset == "standard":
        dataset_class = EEGDataset
    elif config.dataset == "sparse":
        dataset_class = SparseDataset
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")

    dataset = dataset_class(samples_path=Path(config.data_path))
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
    )

    print("Loading ViT...")
    vit = EEGViT.from_config(config.vit)

    print("Loading MAE...")
    mae = EEGViTMAE.from_config(vit, config.mae)

    print("Initialzing optimizer and scheduler..")
    optimizer = torch.optim.AdamW(
        mae.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=config.lr_warmup_steps
    )

    print("Accelerate prepare...")
    mae, optimizer, scheduler, dataloader = accelerator.prepare(
        mae, optimizer, scheduler, dataloader
    )

    trainer = MAETrainer(
        mae=mae, accelerator=accelerator, scheduler=scheduler, optimizer=optimizer
    )

    for epoch in range(config.epochs):
        print(f"Epoch {epoch}/{config.epochs}")
        mae.train()

        i = 0
        for batch in tqdm(dataloader):
            l = trainer.step(batch)
            if i % 100 == 0:
                print(f"Loss: {l['loss']:.3f}")
            i += 1

        checkpoint_dir = Path(config.checkpoint_dir) / config.name / f"epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpointing to {checkpoint_dir}...")
        _model = accelerator.unwrap_model(mae)
        _model.save_pretrained(checkpoint_dir)

        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    final_checkpoint_dir = Path(config.checkpoint_dir) / config.name / "final"
    print(f"Done! Saving to {final_checkpoint_dir}...")

    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _model = accelerator.unwrap_model(mae)
    _model.save_pretrained(final_checkpoint_dir)

    accelerator.end_training()


def main():
    config_file = Path(sys.argv[1])
    config_data = tomllib.loads(config_file.read_text())

    # If name unspecificed, it's the file name without the extension
    config_data["name"] = config_data.get("name", config_file.stem)

    config = Config(**config_data)

    print(f"Training with {config.name}...")
    print(config)

    train(config)


if __name__ == "__main__":
    main()
