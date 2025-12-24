# Builtin imports
import tomllib
from pathlib import Path
import sys
import typing

# Internal imports
from data.dataset import EEGDataset, SparseDataset, SparseClassificationDataset
from models.et import EEGMAE, EEGMAEConfig, MAETrainer
from models.classifiers import (
    EEGClassifier,
    EEGClassifierConfig,
    EEGClassifierTrainer,
)
from models.eegnet import EEGNet, EEGNetConfig, EEGNetTrainer
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

    dataset: (
        typing.Literal["standard"]
        | typing.Literal["sparse"]
        | typing.Literal["sparse_classification"]
        | typing.Literal["aj_preprocessed_classification"]
    ) = "standard"
    class_col: str | None = None

    # Model config
    arch: (
        typing.Literal["mae"] | typing.Literal["classifier"] | typing.Literal["eegnet"]
    ) = "mae"
    mae: EEGMAEConfig | None = None
    classifier: EEGClassifierConfig | None = None
    eegnet: EEGNetConfig | None = None

    # Dataloading
    num_workers: int = 8
    batch_size: int = 256
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

    # Allow to run independently or with torchrun/accelerate launch
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if rank == 0:
        print(f"Loading dataset/dataloader from {config.data_path}...")
    dataset_kwargs = {}

    if config.dataset == "standard":
        dataset_class = EEGDataset
    elif config.dataset == "sparse":
        dataset_class = SparseDataset
    elif config.dataset == "sparse_classification":
        dataset_class = SparseClassificationDataset
        assert config.class_col is not None, (
            "need classifier col to train sparse classification"
        )
        dataset_kwargs["class_col"] = config.class_col
    elif config.dataset == "aj_preprocessed_classification":
        assert config.class_col is not None, (
            "need classifier col to train sparse classification"
        )
        dataset_kwargs["class_col"] = config.class_col
        dataset_kwargs["split"] = "train"

        from data.alljoined.preprocessed import AJPreprocessedClassificationDataset

        dataset_class = AJPreprocessedClassificationDataset
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")

    dataset = dataset_class(Path(config.data_path), **dataset_kwargs)
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
    )

    if rank == 0:
        print(f"Loading {config.arch} model...")
    if config.arch == "mae":
        assert config.mae is not None, "need MAE config to train MAE"
        model = EEGMAE.from_config(config.mae)
    elif config.arch == "classifier":
        assert config.classifier is not None, "need class col to train classifier"
        model = EEGClassifier.from_config(config.classifier)
    elif config.arch == "eegnet":
        assert config.eegnet is not None, "need EEGNet config to train EEGNet"
        model = EEGNet.from_config(config.eegnet)
    else:
        raise ValueError(f"Unknown arch {config.arch}")

    if rank == 0:
        print("Initialzing optimizer and scheduler..")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=config.lr_warmup_steps
    )

    if rank == 0:
        print("Accelerate prepare...")
    model, optimizer, scheduler, dataloader = accelerator.prepare(
        model, optimizer, scheduler, dataloader
    )

    if config.arch == "mae":
        trainer = MAETrainer(
            mae=model, accelerator=accelerator, scheduler=scheduler, optimizer=optimizer
        )
    elif config.arch == "classifier":
        trainer = EEGClassifierTrainer(
            classifier=model,
            accelerator=accelerator,
            scheduler=scheduler,
            optimizer=optimizer,
        )
    elif config.arch == "eegnet":
        trainer = EEGNetTrainer(
            classifier=model,
            accelerator=accelerator,
            scheduler=scheduler,
            optimizer=optimizer,
        )
    else:
        raise ValueError(f"Unknown arch {config.arch}")

    accelerator.wait_for_everyone()

    batch = None
    for f in dataloader:
        batch = f
        print(f"Assigned batch to {f}")
        break

    for epoch in range(config.epochs):
        if rank == 0:
            print(f"Epoch {epoch}/{config.epochs}")
        model.train()

        i = 0
        for _ in tqdm(range(10 * 1000), desc="Trying to overfit"):
            if config.arch == "mae":
                l = trainer.step(batch)
            elif config.arch == "classifier":
                samples, classes = batch
                l = trainer.step(samples, classes)
            elif config.arch == "eegnet":
                samples, classes = batch
                l = trainer.step(samples, classes)
            else:
                raise ValueError(f"bad arch {config.arch}")

            if i % 10 == 0:
                if rank == 0:
                    if config.arch in ("classifier", "eegnet"):
                        print(
                            f"Loss: {l['loss']:.3f} | Accuracy: {l['accuracy'] * 100:.2f}% ({l['num_correct']}/{l['total']})"
                        )
                    else:
                        print(f"Loss: {l['loss']:.3f}")
            i += 1

        checkpoint_dir = Path(config.checkpoint_dir) / config.name / f"epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            print(f"Checkpointing to {checkpoint_dir}...")
        _model = accelerator.unwrap_model(model)
        _model.save_pretrained(checkpoint_dir)

        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    final_checkpoint_dir = Path(config.checkpoint_dir) / config.name / "final"
    print(f"Done! Saving to {final_checkpoint_dir}...")

    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _model = accelerator.unwrap_model(model)
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
