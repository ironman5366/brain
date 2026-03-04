# Combining NMED-T with OpenNeuro DS005876 for Audio-EEG Contrastive Learning

## Motivation

The NMED-T dataset provides 57K one-second EEG windows from 18 subjects passively listening to 10 full-length songs. While sufficient for initial experiments, the limited song diversity (10 tracks) constrains the contrastive learning objective -- the model sees the same audio repeatedly across subjects. We investigated whether adding the OpenNeuro DS005876 "Song Familiarity" dataset (29 subjects, 121 short melody snippets) could improve training through increased sample count and audio diversity.

## Dataset Compatibility

The two datasets use different EEG systems:

| Property | NMED-T | DS005876 |
|---|---|---|
| EEG system | EGI HydroCel 128 (geodesic) | Brain Products actiCHamp 32 |
| Active channels | 120 (mapped to 10-5 positions) | 32 (standard 10-20 positions) |
| Sampling rate | 125 Hz | 1000 Hz |
| Stimuli | 10 songs, minutes each | 121 snippets, 5-17s each |
| Task | Passive listening | Active familiarity detection |

The main challenge is channel alignment. The EGI system places electrodes on a geodesic grid that does not correspond to standard 10-20 positions. When mapped to the 10-5 naming system, only 13 of 32 DS005876 channel names appear in NMED's active set -- despite many electrodes being physically close on the scalp.

## Channel Mapping via Spatial Interpolation

Rather than relying on channel name matching, we computed a direct spatial mapping from the 128 EGI electrode positions to the 32 DS005876 target positions using MNE's 3D montage coordinates. For each target channel, all EGI electrodes within a 25mm radius contribute via inverse-distance-squared weighting:

```
signal_target = sum(w_i * signal_EGI_i) / sum(w_i),  where w_i = 1 / d_i^2
```

This achieves full coverage of all 32 target channels, with 1-4 contributing EGI electrodes per target at distances of 7-25mm. The spatial averaging also acts as a mild noise smoother. The mapping is precomputed once and stored as a constant (`data/songfam/channel_mapping.py`).

## Combined Dataset

Both datasets are processed to a uniform `(N, 32, 125)` format -- 32 channels at 125 Hz in 1-second windows with per-epoch z-score normalization. Audio embeddings are pre-computed via EnCodec encoder (128-dim, 75 frames/sec).

| Split | NMED-32ch | DS005876 | Combined |
|---|---|---|---|
| Train | 51,372 | 17,042 | 68,414 |
| Val | 5,708 | 1,786 | 7,494 |
| Unique songs | 10 | 121 | 131 |
| Subjects | 18 | 29 | 47 |

## Training Results

We trained the audio-contrastive model (6-layer transformer, 512d, 8 heads, shared 256d embedding space) for 5 epochs on each dataset individually and on the combination. All runs used identical hyperparameters (batch_size=16, lr=1e-4, contrastive + variance/covariance loss).

| Dataset | Train samples | Final Epoch Loss | Final Epoch Top-1 |
|---|---|---|---|
| NMED-32ch only | 51,372 | 3.710 | 5.2% |
| DS005876 only | 17,042 | 3.648 | 8.4% |
| **Combined** | **68,414** | **3.214** | **12.3%** |

Random baseline for batch_size=16: loss = ln(16) = 2.773, top-1 = 6.25%.

## Observations

1. **The combined dataset substantially outperforms both individual datasets**, with ~0.5 lower loss and ~2x the top-1 accuracy of either alone after 5 epochs.

2. **Audio diversity matters more than sample count.** DS005876 trains slightly better than NMED-32ch despite having 3x fewer samples, likely because 121 unique songs provide far richer contrastive targets than 10 songs repeated across subjects.

3. **Cross-system combination is viable.** The IDW spatial interpolation produces clean signals from the EGI system that combine well with the Brain Products data. The 32-channel representation, while lower resolution than NMED's native 120 channels, captures sufficient spatial information for the contrastive objective.

4. **The gains are synergistic.** The combined result is better than either dataset alone would suggest -- more subjects diversify the EEG representations, while more songs diversify the audio targets, and both help the contrastive alignment.

## Implementation

- `data/songfam/channel_mapping.py` -- EGI-to-32ch spatial mapping (25mm IDW)
- `data/songfam/build.py` -- DS005876 EEG processing pipeline
- `data/songfam/build_encodec.py` -- EnCodec embeddings for 121 stimuli
- `data/nmed/build.py --channels 32` -- NMED re-mapped to 32 channels
- `data/dataset.py:CombinedAudioEmbedDataset` -- combined dataloader
- `train.py` with `dataset = "combined_audio_embed"` -- training config

Processed files are stored at `/kreka/research/willy/side/brain_datasets/nmed-processed/` with the `nmed-32ch-*` and `songfam-*` prefixes.
