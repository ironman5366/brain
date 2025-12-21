# Builtin imports
from pathlib import Path

# Internal imports
from data.dataset import SparseClassificationDataset
from models.et import EEGMAE
from models.linear_classifier import (
    LinearClassifier,
    LinearClassifierConfig,
    LinearClassifierTrainer,
)

# External imports
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ALLJOINED_BASE_PATH = Path(
    "/kreka/research/willy/side/brain_datasets/alljoined-epochs-2025-12-21/"
)

TRAIN_PATH = ALLJOINED_BASE_PATH / "alljoined-epochs-2025-12-21-train.safetensors"
VAL_PATH = ALLJOINED_BASE_PATH / "alljoined-epochs-2025-12-21-val.safetensors"

CHECKPOINT_PATH = Path.cwd() / "checkpoints"

DEVICE = "cuda:0"


class EvalModel:
    input_dim: int = None

    def transform(samples):
        raise NotImplementedError()


class MeanPoolIdentity(EvalModel):
    input_dim = 308

    def transform(self, samples):
        return torch.mean(samples, dim=1)


class MAEModel(EvalModel):
    def __init__(self, checkpoint_path: Path):
        print(f"loading model from {checkpoint_path}")

        self.model = EEGMAE.from_pretrained(checkpoint_path).to(DEVICE)
        self.model.eval()

        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False

        self.input_dim = self.model.encoder_dim


class MeanPoolMAE(MAEModel):
    def transform(self, samples):
        out = self.model.inference(samples)
        # print("out shape", out.shape)
        return torch.mean(out, dim=1)


def classify():
    # model = MeanPoolMAE(
    #     checkpoint_path=CHECKPOINT_PATH / "aj-epochs-channel-1024-mask-75" / "final"
    # )
    model = MeanPoolMAE(
        checkpoint_path=CHECKPOINT_PATH / "aj-epochs-sample-1024-mask-75" / "epoch_4"
    )
    # model = MeanPoolIdentity()
    # Train
    ds = SparseClassificationDataset(TRAIN_PATH, class_col="category_num")
    class_dim = ds.class_dim

    val_ds = SparseClassificationDataset(VAL_PATH, class_col="category_num")
    assert val_ds.class_dim == ds.class_dim, (
        "Val and train DS have mismatched classification columns"
    )

    dl = DataLoader(ds, num_workers=8, batch_size=1024, shuffle=True)
    conf = LinearClassifierConfig(input_dim=model.input_dim, num_classes=class_dim)
    classifier = LinearClassifier.from_config(conf).to(DEVICE)
    classifier.train()

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
    )
    trainer = LinearClassifierTrainer(classifier=classifier, optimizer=optimizer)
    i = 0

    for batch in tqdm(dl, desc="Training 1 epoch of classifier..."):
        samples, classes = batch

        samples = samples.to(DEVICE)
        classes = classes.to(DEVICE)

        samples = model.transform(samples)

        loss = trainer.step(samples, classes)["loss"]

        if i % 10 == 0:
            print(f"Loss: {loss}")

        i += 1

    del ds
    del dl

    with torch.inference_mode():
        val_dl = DataLoader(val_ds, num_workers=8, batch_size=1024, shuffle=True)

        i = 0

        all_acc = None

        for batch in tqdm(val_dl):
            samples, classes = batch

            samples = model.transform(samples.to(DEVICE))
            classes = classes.to(DEVICE)

            out = classifier(samples)
            preds = out.argmax(dim=-1)
            accuracy = classes == preds

            if all_acc is None:
                all_acc = accuracy
            else:
                all_acc = torch.cat((all_acc, accuracy))

            if i % 100 == 0:
                print(
                    f"Accuracy: {len(torch.nonzero(accuracy)) / len(accuracy):.5%}, overall {len(torch.nonzero(all_acc)) / len(all_acc):.5%}"
                )

            i += 1

        nonzero = len(torch.nonzero(all_acc))
        total = len(all_acc)
        perc = nonzero / total
        print(f"Overall: {nonzero:,}/{total:,} correct, {perc:.0%}")


if __name__ == "__main__":
    classify()
