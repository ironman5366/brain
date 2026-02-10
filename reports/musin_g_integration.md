# Adding MUSIN-G (OpenNeuro ds003774) to Audio-EEG Contrastive Learning

## Motivation

The previous experiment showed that combining NMED-T (10 songs, 18 subjects) with DS005876 Song Familiarity (121 snippets, 29 subjects) substantially improved contrastive learning -- loss 3.214 vs 3.710/3.648 individually, and 12.3% top-1 vs 5.2%/8.4%. Audio diversity was the biggest driver.

MUSIN-G (ds003774) is a natural next dataset to add because it uses the **same EGI HydroCel 128-channel system** as NMED-T, meaning the existing IDW spatial interpolation can be reused directly with zero additional mapping work. It provides 12 diverse-genre songs (deep house, indie, ambient, Hindustani classical, goth rock, etc.) listened to by 20 subjects.

## Dataset Properties

| Property | NMED-T | DS005876 | MUSIN-G |
|---|---|---|---|
| EEG system | EGI HydroCel 128 | Brain Products 32 | EGI HydroCel 128 |
| Active channels | 120 (mapped to 32) | 32 native | 125 (mapped to 32) |
| Sampling rate | 125 Hz (native) | 1000 Hz (downsampled) | 250 Hz (downsampled) |
| Subjects | 18 | 29 | 20 |
| Stimuli | 10 songs (~3-5 min) | 121 snippets (5-17s) | 12 songs (~100-130s) |
| Task | Passive listening | Active familiarity detection | Passive listening (eyes closed) |
| Behavioral data | -- | Familiarity detection | Enjoyment + familiarity (1-5) |
| Data format | BIDS (.set) | BIDS (.set) | BIDS (.set) |

### MUSIN-G Songs

| ID | Title | Artist | Genre | Duration |
|---|---|---|---|---|
| 1 | Albela Sajan | Kaushiki Chakraborty | Hindustani Classical | 120s |
| 2 | Trip to the Lonely Planet | Ozric Tentacles | Space Rock | 113s |
| 3 | Bela Lugosi's Dead | Bauhaus | Goth Rock | 130s |
| 4 | Duvet | Boa | Dream Pop | 112s |
| 5 | 10 Mile Stereo | Beach House | Indie | 120s |
| 6 | A Walk | Tycho | Ambient | 120s |
| 7 | Sail | Awolnation | Indie Rock | 108s |
| 8 | The Ministry of Lost Souls | Dream Theater | Progressive Metal | 130s |
| 9 | The Other Side | Pendulum | Drum and Bass | 106s |
| 10 | Eple | Royksopp | Electronic | 118s |
| 11 | Insomnia | Faithless | Deep House | 120s |
| 12 | Jai Ho | A.R. Rahman | Soft Jazz | 113s |

## Processing Pipeline

### EEG Processing (`data/musin_g/build.py`)

1. For each of 20 subjects x 12 sessions (one song per session):
   - Load BIDS `.set` file via `mne.io.read_raw_eeglab()`
   - Parse `events.tsv` to find music onset (`stim`/`stm+`) and offset (`opyp`/`fxnd`) markers
   - Filter events to those within the recording's sample range (events files contain the full experiment log)
   - Extract the music-listening EEG segment
   - Pick channels E1-E124 + E129 (renamed to Cz); drop E125-E128 (face/neck electrodes)
   - Downsample 250 Hz -> 125 Hz
   - Apply `map_egi_to_32ch()` using existing `EGI_TO_32CH_MAP` from `data/songfam/channel_mapping.py`
   - Window into 1-second epochs with per-epoch z-score normalization

2. **Offset marker fallback**: Some sessions (primarily subjects 16-20) lack the `opyp`/`fxnd` offset markers because the event timestamps exceed the recording's sample range. In these cases, the pipeline falls back to `onset + known_song_duration`, which correctly captures the music segment.

3. Subject-level 90/10 train/val split (seed=42, same strategy as other datasets).

### Audio Embeddings (`data/musin_g/build_encodec.py`)

- 12 WAV files from `Code/ESongs/` encoded through Facebook EnCodec (24kHz, 128-dim, 75 frames/sec)
- Windowed into 1-second chunks aligned with EEG window indices
- 52 of 27,860 samples (<0.2%) have window indices slightly exceeding audio duration -- zero-filled

### Processing Results

| Split | NMED-32ch | DS005876 | MUSIN-G | Combined |
|---|---|---|---|---|
| Train | 40,539 | 27,875 | 25,075 | 93,489 |
| Val | 5,708 | 1,786 | 2,785 | 10,279 |
| Unique songs | 10 | 121 | 12 | 143 |
| Subjects | 18 | 29 | 20 | 67 |

All datasets stored as `(N, 32, 125)` tensors with per-epoch z-normalization.

## Training Results

Same architecture and hyperparameters as the previous experiment: 6-layer transformer encoder, 512d, 8 heads, 256d shared embedding space, symmetric InfoNCE + VICReg regularization. Trained for 5 epochs with batch_size=16, lr=1e-4, 200 warmup steps.

### Per-Epoch Comparison

**Three-dataset (NMED + songfam + MUSIN-G) -- 93,489 train samples:**

| Epoch | Avg Loss | Avg Top-1 | Start Loss | End Loss |
|---|---|---|---|---|
| 0 | 3.083 | 17.1% | 3.349 | 2.960 |
| 1 | 2.910 | 18.0% | 2.971 | 2.864 |
| 2 | 2.850 | 18.6% | 2.860 | 2.823 |
| 3 | 2.842 | 18.1% | 2.811 | 2.837 |
| 4 | 2.835 | 18.3% | 2.823 | 2.873 |

**Two-dataset (NMED + songfam) -- 68,414 train samples:**

| Epoch | Avg Loss | Avg Top-1 | Start Loss | End Loss |
|---|---|---|---|---|
| 0 | 3.337 | 12.0% | 3.482 | 3.221 |
| 1 | 3.270 | 12.0% | 3.317 | 3.292 |
| 2 | 3.252 | 12.5% | 3.287 | 3.224 |
| 3 | 3.222 | 12.8% | 3.245 | 3.220 |
| 4 | 3.245 | 12.6% | 3.238 | 3.228 |

### Summary

| Configuration | Train Samples | Final Loss | Final Top-1 | vs Random |
|---|---|---|---|---|
| Two-dataset (NMED + songfam) | 68,414 | 3.245 | 12.6% | 2.0x |
| **Three-dataset (+ MUSIN-G)** | **93,489** | **2.835** | **18.3%** | **2.9x** |
| Random baseline (batch=16) | -- | 2.773 | 6.25% | 1.0x |

Adding MUSIN-G reduced loss by **12.6%** (3.245 -> 2.835) and improved top-1 accuracy by **45%** relative (12.6% -> 18.3%).

## Observations

1. **MUSIN-G integration required minimal effort due to hardware compatibility.** The same EGI HydroCel 128-channel system means the existing `EGI_TO_32CH_MAP` spatial interpolation works directly. The only new work was parsing the BIDS events structure and handling the offset marker fallback.

2. **The improvement is substantial despite only 12 new songs.** The 12 MUSIN-G songs span very diverse genres (Hindustani classical, goth rock, drum and bass, ambient, etc.), which likely provides more distinctive contrastive targets than adding more similar songs would.

3. **The three-dataset model approaches the theoretical floor.** With a final loss of 2.835 vs the random baseline of ln(16) = 2.773, the model is only 0.062 nats above the information-theoretic minimum. This suggests the batch size may now be the bottleneck -- larger batches would provide harder negatives and more room for improvement.

4. **Training converges faster.** The three-dataset model reaches loss ~3.0 within epoch 0, while the two-dataset model doesn't reach that level even after 5 epochs.

5. **A training stability bug was identified and fixed.** The original code had a subtle issue where the last batch of an epoch could contain a single sample (93,489 % 16 = 1). The VICReg variance loss (`std(dim=0)` with Bessel's correction on N=1) and covariance loss (division by N-1=0) produced NaN, corrupting all subsequent training. Fixed by adding `drop_last=True` to the DataLoader and adding batch-size guards to the loss functions.

## Next Steps

- **Increase batch size**: The model is near the ln(batch_size) floor. Training with batch_size=32 or 64 would provide harder negatives and more headroom.
- **Add more datasets**: The pipeline now supports arbitrary numbers of combined datasets. Other EGI-based EEG-music datasets could be added with minimal effort.
- **Evaluate on downstream tasks**: The improved representations should be evaluated on tasks like genre classification, familiarity prediction, or cross-modal retrieval.

## Implementation

- `data/musin_g/songs.py` -- dataset constants (subjects, songs, paths)
- `data/musin_g/build.py` -- EEG processing pipeline with event-based segmentation
- `data/musin_g/build_encodec.py` -- EnCodec audio embeddings for 12 songs
- `configs/nmed/contrastive_three_dataset.toml` -- three-dataset training config
- `train.py` -- extended to support 3rd dataset via `data_path_3`/`audio_embeds_path_3`

Processed files at `/kreka/research/willy/side/brain_datasets/nmed-processed/` with `musin-g-*` prefix.
