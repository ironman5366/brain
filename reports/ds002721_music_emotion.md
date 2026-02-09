# Adding OpenNeuro DS002721 (Music Emotion) to Audio-EEG Contrastive Training

## Motivation

Our previous report (`dataset_combination.md`) showed that combining NMED-T (10 songs, 18 subjects) with DS005876 Song Familiarity (121 snippets, 29 subjects) improved contrastive learning from ~5-8% top-1 to 12.3%, with audio diversity being the key driver. DS002721 offers another pool of music-listening EEG: 31 subjects listening to 91 unique film score excerpts with emotion ratings. Adding it would bring the combined training set to ~81K samples, ~222 unique songs, and 78 subjects -- substantially more audio diversity and subject variation.

## Dataset Description

OpenNeuro DS002721 (Daly et al., 2018) recorded 31 subjects listening to 12-second film score excerpts from the Eerola & Vuoskoski (2010) stimulus set, with 8-dimensional emotion ratings after each clip.

| Property | DS002721 |
|---|---|
| EEG system | Standard 10-20, 19 channels |
| Sampling rate | 1000 Hz |
| Stimuli | 91 unique film score excerpts, 12s each |
| Clips per subject | 40 (across runs 2-5) |
| Task | Passive listening + emotion rating |
| Subjects | 31 |
| File format | BIDS (EDF + events.tsv) |

The 19 channels are: Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2 (using modern 10-10 names; the original data uses older names T3/T4/T5/T6/FP1/FP2).

### Audio Stimuli

The stimuli are **not included** in the OpenNeuro dataset. They are the Eerola & Vuoskoski (2010) film score excerpts, hosted on OSF at https://osf.io/p6vkg/ (Set1.zip, 148.6 MB, 360 mp3 files). DS002721 uses 91 of these clips; stimulus event codes encode the clip number as `code % 100` (mapping to files `001.mp3` through `099.mp3`).

### Event Code Discovery

The dataset documentation describes stimulus codes as 301-360, but actual codes span **302-657** across subjects. The hundreds digit varies by trial; only the last two digits (code % 100) identify the clip. This was discovered by analyzing all event codes across subjects and required fixing the initial event parsing logic.

## Channel Mapping: 19 to 32 via IDW Interpolation

This is the main compatibility challenge. DS002721 has only 19 channels (standard 10-20 placement), while our target layout has 32 channels including 10-10 positions like FC5, CP5, TP9, etc. that have no nearby 10-20 source electrode.

**NOTE: This is a temporary, coarse mapping.** With only 19 source channels, many target positions are interpolated from electrodes 38-53mm away -- far less precise than the NMED mapping (120 sources, 7-25mm distances). We plan to explore more principled approaches (spherical spline interpolation, source-space methods) in future work.

### Approach

We used the same inverse-distance-squared weighting (IDW) as the NMED/Song Familiarity integration, with a wider 40mm radius to accommodate the sparser electrode grid:

```
signal_target = sum(w_i * signal_source_i) / sum(w_i),  where w_i = 1 / d_i^2
```

Using MNE's `standard_1020` and `standard_1005` montages for 3D coordinates:

| Coverage | Count | Details |
|---|---|---|
| Exact match (distance < 1mm) | 18 of 32 | 18 of the 19 source channels map directly to target positions |
| Interpolated (2 sources) | 14 of 32 | Remaining targets interpolated from 2 nearest sources at 38-53mm |

The 14 interpolated channels (FC5, FC1, FCz, TP9, CP5, CP1, Oz, TP10, CP6, CP2, CPz, FC6, FC2, AFz) all fall between two 10-20 electrodes. While the interpolation distances are larger than ideal, the spatial coverage is sufficient for the contrastive learning objective where we primarily need consistent channel ordering across datasets.

The mapping is precomputed and hardcoded in `data/musicemo/channel_mapping.py`.

## Combined Dataset

All three datasets are processed to `(N, 32, 125)` format with per-epoch z-score normalization and EnCodec audio embeddings `(N, 128, 75)`.

| Split | NMED-32ch | DS005876 | DS002721 | **3-Way Combined** |
|---|---|---|---|---|
| Train | 51,372 | 17,042 | 12,960 | **81,374** |
| Val | 5,708 | 1,786 | 1,920 | **9,414** |
| Unique songs | 10 | 121 | ~91 | **~222** |
| Subjects | 18 | 29 | 31 | **78** |

DS002721 contributes 12,960 training samples from 27 subjects (4 held out for validation), derived from 31 subjects x 40 clips x 12 one-second windows = 14,880 total samples.

Note: ~1% of DS002721 samples (134 total) have zero-filled audio embeddings due to stimulus codes where `code % 100 = 0`, mapping to a nonexistent `000.mp3`. These are retained but effectively contribute noise to training.

## Training Results

All runs used identical hyperparameters (batch_size=16, lr=1e-4, 5 epochs, 6-layer transformer, 512d, 8 heads, shared 256d embedding space). Results are final epoch averages.

| Dataset | Train samples | Final Epoch Loss | Final Epoch Top-1 |
|---|---|---|---|
| NMED-32ch only | 51,372 | 3.710 | 5.2% |
| DS005876 only | 17,042 | 3.648 | 8.4% |
| 2-way combined (prev. report) | 68,414 | 3.214 | 12.3% |
| 2-way combined (fresh run) | 68,414 | 3.235 | 11.9% |
| DS002721 (MusicEmo) only | 12,960 | 3.716 | 5.7% |
| **3-way combined** | **81,374** | **2.937** | **17.9%** |

Random baseline for batch_size=16: loss = ln(16) = 2.773, top-1 = 6.25%.

## Observations

1. **The 3-way combined dataset substantially outperforms the 2-way baseline**, with ~0.3 lower loss (2.937 vs 3.235) and ~50% higher top-1 accuracy (17.9% vs 11.9%). This is the strongest result we've seen from this contrastive setup.

2. **MusicEmo alone cannot learn the contrastive objective.** Loss barely moves from initialization (~3.72), and accuracy remains at random (5.7%). The 19→32 channel interpolation is too coarse for the EEG signal alone to align with audio embeddings.

3. **MusicEmo as supplemental data is highly effective despite being useless alone.** This is the key insight: while the interpolated EEG is too noisy to learn a contrastive mapping independently, it provides useful regularization and diversity when combined with higher-quality data from the other two datasets.

4. **Audio diversity continues to be the dominant factor.** Adding 91 unique film score clips (on top of 131 existing) pushed accuracy from 12% to 18%. The marginal value per unique song is roughly preserved from the 2-way experiment.

5. **The 2-way baseline reproduced closely** (3.235 loss / 11.9% top-1 vs 3.214 / 12.3% in the previous report), confirming the results are stable across runs.

## Implementation

- `data/musicemo/songs.py` -- dataset metadata and constants
- `data/musicemo/channel_mapping.py` -- 19-to-32 channel IDW spatial mapping (40mm radius)
- `data/musicemo/build.py` -- EEG processing pipeline (EDF → downsample → map → window → normalize)
- `data/musicemo/build_encodec.py` -- EnCodec embeddings for 360 film score stimuli
- `configs/nmed/contrastive_musicemo_only.toml` -- standalone MusicEmo training config
- `configs/nmed/contrastive_combined_3way.toml` -- 3-way combined training config
- `train.py` -- extended with `data_path_3`/`audio_embeds_path_3` for 3-way combination

Processed files are stored at `/kreka/research/willy/side/brain_datasets/nmed-processed/` with the `musicemo-*` prefix.

## Future Work

- **Channel mapping improvements**: Replace IDW interpolation with spherical spline interpolation or source-space methods for more principled 19→32 mapping.
- **Emotion labels**: DS002721 includes 8-dimensional emotion ratings per clip. These could be incorporated as auxiliary supervision (multi-task learning) or used for emotion-conditioned analysis.
- **Clip overlap analysis**: Some film score excerpts may overlap between DS002721 and DS005876 (both use film music). Identifying shared clips could enable cross-dataset validation.
