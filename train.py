# Builtin imports
import tomllib
from pathlib import Path
import sys
import typing

# Internal imports
from data.dataset import EEGDataset, SparseDataset, SparseClassificationDataset
from models.et import EEGMAE, EEGMAEConfig, MAETrainer
from models.big_classifier import (
    EEGClassifier,
    EEGClassifierConfig,
    EEGClassifierTrainer,
)
from models.ccnn import (
    CCNN1DConfig,
    CCNN1DRegressor,
    CCNN1DRegressorTrainer,
    CCNN1DClassifierConfig,
    CCNN1DClassifier,
    CCNN1DClassifierTrainer,
)
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
        | typing.Literal["deap_regression"]
        | typing.Literal["deap_classification"]
    ) = "standard"
    class_col: str | None = None
    label_col: str | None = None  # For regression datasets

    # Model config
    arch: (
        typing.Literal["mae"]
        | typing.Literal["classifier"]
        | typing.Literal["ccnn_regressor"]
        | typing.Literal["ccnn_classifier"]
    ) = "mae"
    mae: EEGMAEConfig | None = None
    classifier: EEGClassifierConfig | None = None
    ccnn: CCNN1DConfig | None = None
    ccnn_cls: CCNN1DClassifierConfig | None = None

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
    elif config.dataset == "deap_regression":
        assert config.label_col is not None, (
            "need label_col for deap regression (e.g., valence, arousal)"
        )
        dataset_kwargs["label_col"] = config.label_col
        dataset_kwargs["split"] = "train"

        from data.deap import DEAPRegressionDataset

        dataset_class = DEAPRegressionDataset
    elif config.dataset == "deap_classification":
        assert config.class_col is not None, (
            "need class_col for deap classification (e.g., valence_high, arousal_high)"
        )
        dataset_kwargs["class_col"] = config.class_col
        dataset_kwargs["split"] = "train"

        from data.deap import DEAPClassificationDataset

        dataset_class = DEAPClassificationDataset
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")

    dataset = dataset_class(Path(config.data_path), **dataset_kwargs)
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
    )

    print(f"Loading {config.arch} model...")
    if config.arch == "mae":
        assert config.mae is not None, "need MAE config to train MAE"
        model = EEGMAE.from_config(config.mae)
    elif config.arch == "classifier":
        assert config.classifier is not None, "need classifier config to train classifier"
        model = EEGClassifier.from_config(config.classifier)
    elif config.arch == "ccnn_regressor":
        assert config.ccnn is not None, "need ccnn config to train ccnn_regressor"
        model = CCNN1DRegressor.from_config(config.ccnn)
    elif config.arch == "ccnn_classifier":
        assert config.ccnn_cls is not None, "need ccnn_cls config to train ccnn_classifier"
        model = CCNN1DClassifier.from_config(config.ccnn_cls)
    else:
        raise ValueError(f"Unknown arch {config.arch}")

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
    elif config.arch == "ccnn_regressor":
        trainer = CCNN1DRegressorTrainer(
            model=model,
            accelerator=accelerator,
            scheduler=scheduler,
            optimizer=optimizer,
        )
    elif config.arch == "ccnn_classifier":
        trainer = CCNN1DClassifierTrainer(
            model=model,
            accelerator=accelerator,
            scheduler=scheduler,
            optimizer=optimizer,
        )
    else:
        raise ValueError(f"Unknown arch {config.arch}")

    for epoch in range(config.epochs):
        print(f"Epoch {epoch}/{config.epochs}")
        model.train()

        i = 0
        for batch in tqdm(dataloader):
            if config.arch == "mae":
                l = trainer.step(batch)
            elif config.arch == "classifier":
                samples, classes = batch
                l = trainer.step(samples, classes)
            elif config.arch == "ccnn_regressor":
                samples, labels = batch
                l = trainer.step(samples, labels)
            elif config.arch == "ccnn_classifier":
                samples, classes = batch
                l = trainer.step(samples, classes)
            else:
                raise ValueError(f"bad arch {config.arch}")

            if i % 100 == 0:
                if config.arch == "classifier" or config.arch == "ccnn_classifier":
                    print(
                        f"Loss: {l['loss']:.3f} | Accuracy: {l['accuracy'] * 100:.2f}% ({l['num_correct']}/{l['total']})"
                    )
                elif config.arch == "ccnn_regressor":
                    print(f"Loss: {l['loss']:.3f} | MAE: {l['mae']:.2f} | Within 1pt: {l['within_1']*100:.1f}% | Binary Acc: {l['binary_acc']*100:.1f}%")
                else:
                    print(f"Loss: {l['loss']:.3f}")
            i += 1

        checkpoint_dir = Path(config.checkpoint_dir) / config.name / f"epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
