# Builtin imports
from pathlib import Path

# External imports
from tqdm import tqdm
import mne
import polars as pl
import ray
import torch
from safetensors.torch import save_file

# Location you've downloaded https://openneuro.org/datasets/ds003825/versions/1.2.0
BASE_PATH = Path(
    "/kreka/research/willy/side/brain_datasets/openneuro_thingseeg_ds003825/"
)
OUT_DIR = Path("/kreka/research/willy/side/brain_datasets/things/")

EXPECTED_CH_NAMES = {
    "Fp1",
    "Fz",
    "F3",
    "F7",
    "FT9",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "TP9",
    "CP5",
    "CP1",
    "Pz",
    "P3",
    "P7",
    "O1",
    "Oz",
    "O2",
    "P4",
    "P8",
    "TP10",
    "CP6",
    "CP2",
    "C4",
    "T8",
    "FT10",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "F1",
    "F5",
    "FT7",
    "FC3",
    "C1",
    "C5",
    "TP7",
    "CP3",
    "P1",
    "P5",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "P6",
    "P2",
    "CPz",
    "CP4",
    "TP8",
    "C6",
    "C2",
    "FC4",
    "FT8",
    "F6",
    "AF8",
    "AF4",
    "F2",
    "FCz",
}


def load_sub(sub: Path):
    eeg = sub / "eeg"
    events_csv = eeg / f"{sub.name}_task-rsvp_events.csv"
    metadata = pl.read_csv(events_csv)

    # Add a column for the subject
    sub_name = sub.name
    sub_id = int(sub_name.split("-")[1])
    metadata = metadata.with_columns(
        pl.lit(sub_id).alias("subject_id"), pl.lit(sub_name).alias("subject_name")
    )

    vhdr = eeg / f"{sub.name}_task-rsvp_eeg.vhdr"
    raw_data = mne.io.read_raw_brainvision(vhdr)

    # Pull out event Event/E 1 / 10001 which is stimulus onsert
    events, event_id = mne.events_from_annotations(raw_data, regexp=".*1.*")

    # Load the 50 milliseconds before (blank), the 50 milliseconds of the image, and the 50 milliseconds of blank after it
    epochs = mne.Epochs(raw_data, events, event_id=event_id, tmin=-0.05, tmax=0.1)
    sub_data = torch.from_numpy(epochs.get_data())
    ch_indices = torch.zeros(sub_data.shape[1], dtype=torch.bool)

    for idx, ch in enumerate(raw_data.ch_names):
        if ch in EXPECTED_CH_NAMES:
            ch_indices[idx] = True

    sub_data = sub_data[:, ch_indices]
    print(f"Loaded {sub_data.shape} from {sub.name}")

    if sub_data.shape[1] != 63:
        print(f"Channel badness with sub {sub}!!!! Shape {sub_data.shape}")
        return None, None

    # TODO: don't drop all this data, actually figure out what's going on
    if len(sub_data) != len(metadata):
        print(
            f"Badness with sub {sub}!!!! Metadata len {len(metadata):,} != data len {len(sub_data):,}"
        )
        return None, None

    # TODO: figure out what's going on with these rows
    # Drop the rows with an objectnumber=-1
    metadata_indexed = metadata.with_row_index()
    good_mask = metadata_indexed["objectnumber"] != -1

    bad_count = (~good_mask).sum()
    if bad_count > 0:
        print(f"Removing {bad_count:,} bad objects from sub {sub_name} data")

    # Get indices of good rows to filter the tensor
    good_indices = torch.zeros(sub_data.shape[0], dtype=torch.bool)
    for row in metadata_indexed.filter(good_mask).iter_rows(named=True):
        good_indices[row["index"]] = True

    # Filter both metadata and tensor
    metadata = metadata.filter(pl.col("objectnumber") != -1)
    sub_data = sub_data[good_indices]

    bad_objects = metadata_indexed.filter(objectnumber=-1)
    if len(bad_objects) > 0:
        print(
            f"Removing {len(bad_objects):,} bad objects from sub {sub_name} data, {good_indices.shape}"
        )

    # Make sure samples == metadata
    assert len(sub_data) == len(metadata), "Uh oh"
    return sub_data, metadata


def build_ds(name: str, subs_limit: int | None):
    print("Starting ray...")
    ray.init()

    subs = list(
        sorted(
            [x for x in BASE_PATH.iterdir() if x.name.startswith("sub-") and x.is_dir()]
        )
    )
    if subs_limit is not None:
        subs = [x for x in subs if int(x.name.split("-")[1]) <= subs_limit]

    print(f"Found {len(subs)} subs (limit {subs_limit})")

    load_sub_remote = ray.remote(num_cpus=8)(load_sub)
    futs = []
    for sub in tqdm(subs):
        futs.append(load_sub_remote.remote(sub))

    all_data: pl.DataFrame = None
    all_metadata: pl.DataFrame = None

    for fut in tqdm(futs, desc="Loading things EEG data..."):
        sub_samples, sub_metadata = ray.get(fut)
        if sub_samples is None:
            print("WARNING: DROPPED DATA!!!")
            continue

        if all_data is None:
            all_data = sub_samples
        else:
            all_data = torch.cat((all_data, sub_samples))

        if all_metadata is None:
            all_metadata = sub_metadata
        else:
            for col in all_metadata.columns:
                if col not in sub_metadata.columns:
                    print(f"Dropping col {col} from all metadata")
                    all_metadata = all_metadata.drop(col)

            for col in sub_metadata.columns:
                if col not in all_metadata.columns:
                    print(f"Dropping col {col} from sub metadata")
                    sub_metadata = sub_metadata.drop(col)

            sub_metadata = sub_metadata.select(all_metadata.columns)
            all_metadata = all_metadata.vstack(sub_metadata)

    # TODO: train/test split
    print(f"Loaded {all_data.shape} data")

    parent = OUT_DIR / name
    parent.mkdir(exist_ok=True, parents=True)

    data_file = parent / f"{name}.safetensors"
    save_file({"samples": all_data}, data_file)
    print(f"Saed samples to {data_file}")

    metadata_file = parent / f"{name}-metadata.parquet"
    all_metadata.write_parquet(metadata_file)

    print(f"Saved metadata to {metadata_file}")


if __name__ == "__main__":
    build_ds("things-5-subs-2025-12-27", subs_limit=5)
