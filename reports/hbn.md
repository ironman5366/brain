# Healthy Brain Network (HBN) Movie Watching EEG Dataset

## Overview

The Healthy Brain Network (HBN) is a large-scale pediatric neuroimaging dataset (ages 5-21) collected by the Child Mind Institute. Among its six EEG tasks, the **Movie Watching** condition recorded EEG while participants viewed four short video clips containing complex audio (dialogue, music, sound effects). With **2,639 subjects** and **~1.7 million** potential 1-second windows, this dwarfs both NMED-T (~57K windows, 18 subjects) and DS005876 (~19K windows, 29 subjects) by roughly two orders of magnitude.

The audio content is naturalistic rather than pure music -- this is both the opportunity (diverse audio-EEG pairs, real-world processing) and the open question (how well does it combine with music-only datasets for contrastive learning?).

## Data Location

- **BIDS EEG** (primary, use this): `/kreka/research/willy/side/brain_datasets/HBN/BIDS_EEG/`
  - 10 releases: `cmi_bids_R1` through `cmi_bids_R9` + `cmi_bids_NC`
  - 2,639 total subjects across all releases
  - Per-subject files in EEGLAB `.set` format, one per task
  - BIDS-compliant with `_events.tsv`, `_channels.tsv`, `_eeg.json`, `_electrodes.tsv`
  - Task names already resolved from randomized presentation order

- **Raw EEG** (alternate): `/kreka/research/willy/side/brain_datasets/HBN/EEG/`
  - Per-subject dirs with `raw/` and `preprocessed/` subdirs
  - Video files named by presentation position (`Video1.mat`, `Video2.mat`, `Video3.mat`) -- requires `movie_order.mat` to resolve which movie is which
  - The BIDS version is preferred since it already resolves this

- **Video clips**: `/kreka/research/willy/side/brain_datasets/hbn-video/`
  - `Three Little Kittens- Despicable Me [HNXxJIhVALI].mp4`
  - `Fun with Fractals [XwWyTts06tU].webm`
  - `Diary of a Wimpy Kid Trailer [7ZVEIgPeDCE].webm`
  - `The Present [152985022].mp4`

## EEG Recording Specs

| Property | Value |
|---|---|
| System | EGI HydroCel GSN-128 (same as NMED-T) |
| Channels | 129 (E1-E128 + Cz reference) |
| Sampling rate | 500 Hz (downsample to 125 Hz for pipeline) |
| Reference | Cz |
| Format | EEGLAB `.set` (load via `mne.io.read_raw_eeglab`) |
| Channel naming | E1-E128 (EGI numbering, not 10-20 names) |

Because this is the same EGI system as NMED-T, the existing `EGI_TO_32CH_MAP` IDW spatial interpolation can be reused directly for 32-channel mapping.

## The Four Movie Clips

The protocol presented four short video clips. The first three were shown in randomized order (recorded in `block_perm`); "The Present" was always shown last.

| BIDS Task Name | Clip | Source | Full Video Duration | Trim Applied | EEG Stimulus Duration |
|---|---|---|---|---|---|
| `task-DespicableMe` | "Three Little Kittens" scene from Despicable Me | YouTube `HNXxJIhVALI` | 170.64s | None (full clip) | 170.55s |
| `task-FunwithFractals` | Fun with Fractals educational video | YouTube `XwWyTts06tU` | 303.53s | 0:08 to 2:51 | 163.00s |
| `task-DiaryOfAWimpyKid` | Diary of a Wimpy Kid movie trailer | YouTube `7ZVEIgPeDCE` | 117.49s | None (full clip) | 117.40s |
| `task-ThePresent` | "The Present" animated short | Vimeo `152985022` | 258.62s | Start to credits (~3:23) | 203.07s |

### Duration Verification

Stimulus durations (computed as `video_stop - video_start` from the BIDS events files) are consistent across all subjects to within 0.01s:

| Movie | R1/NDARAC904DMU | R2/NDARAB793GL3 | R3/NDARAA948VFH |
|---|---|---|---|
| DespicableMe | 170.54s | 170.55s | 170.55s |
| FunwithFractals | 162.99s | 163.00s | 162.99s |
| DiaryOfAWimpyKid | 117.39s | 117.40s | 117.40s |
| ThePresent | 203.07s | 203.07s | 203.06s |

The `video_start` times vary per subject (1-4s of pre-stimulus padding) but the stimulus duration itself is locked. This means all subjects' EEG at window N corresponds to the same audio frame.

### Audio Trim Points for the Build Pipeline

- **Despicable Me**: Use full audio track. Residual of ~0.09s vs file duration is within 1-window tolerance.
- **Fun with Fractals**: Extract audio from `t=8s` to `t=171s` (163s). The full video is 303s but only this segment was shown.
- **Diary of a Wimpy Kid**: Use full audio track. Same ~0.09s residual.
- **The Present**: Extract audio from `t=0s` to `t=203.07s`. Visual inspection confirms: at 200s the story is still playing (boy walking out door), at 203s the final scene (empty doorway), and at 206s the title card / credits begin.

## BIDS Event Structure

Each task's `_events.tsv` has a simple format:

```
onset    duration  sample   value         event_code
0        n/a       0        9999          9999
1.04     n/a       520      video_start   83
171.584  n/a       85792    video_stop    103
172.093  n/a       86046.5  boundary      boundary
```

Event codes per movie: DespicableMe = 81/101 (start/stop) or 83/103, FunwithFractals = 82/102, DiaryOfAWimpyKid = 81/101, ThePresent = 84/104. The reliable approach is to match on the `value` field (`video_start`/`video_stop`) rather than numeric codes.

## Data Quality

The `participants.tsv` in each release includes per-task QC flags:

- `available` -- data is usable
- `caution` -- data may have quality issues
- `unavailable` -- data is missing or unusable

The build pipeline should filter to `available` only. Additional columns include `age`, `sex`, `ehq_total` (handedness), and psychiatric factor scores (`p_factor`, `attention`, `internalizing`, `externalizing`).

## Scale Comparison

| Property | NMED-T | DS005876 | HBN (Movie Watching) |
|---|---|---|---|
| Subjects | 18 | 29 | 2,639 |
| Stimuli | 10 songs | 121 snippets | 4 video clips |
| Stimulus duration (total) | ~15 min | ~14 min | ~10.9 min |
| 1-second windows/subject | ~3,170 | ~655 | ~654 |
| Total 1-second windows | ~57K | ~19K | **~1.73M** |
| EEG system | EGI 128ch | Brain Products 32ch | EGI 128ch |
| Native sampling rate | 125 Hz | 1000 Hz | 500 Hz |
| Audio content | Music | Music | Mixed (dialogue, music, SFX) |

## Key Considerations for Integration

1. **Shared stimulus across all subjects.** Unlike NMED (10 songs x 18 subjects) or DS005876 (121 snippets per subject), here all 2,639 subjects watch the exact same 4 clips. This means the audio embedding for window N is identical across subjects -- the contrastive objective will need to account for this. Options include within-subject negative sampling, or treating the massive subject count as a feature for learning robust EEG-audio alignment.

2. **Audio content is not pure music.** The clips contain dialogue, sound effects, music, and silence in varying proportions. This is different from the music-only datasets but may help the model generalize to broader audio-neural alignment.

3. **Pediatric population.** Subjects are 5-21 years old, potentially with different neural signatures than the adult populations in NMED/DS005876. This could be a confound or a feature for robustness.

4. **Scale requires batched processing.** 2,639 subjects with 500 Hz 129-channel EEG files will need parallel/batched processing (Ray or similar). Each release can be processed independently.

## Implementation Plan

Following the existing pattern in `data/nmed/build.py` and `data/songfam/build.py`:

1. **`data/hbn/build.py`** -- Load BIDS `.set` files via MNE, extract `video_start` to `video_stop` segments, downsample 500 -> 125 Hz, apply EGI-to-32ch IDW mapping, window into 1-second epochs, normalize, save safetensors + parquet metadata.

2. **`data/hbn/build_encodec.py`** -- Extract audio from 4 video clips (with trimming), generate EnCodec embeddings windowed to match EEG windows. Only 4 audio files to process (vs per-subject), so this is fast.

3. **`data/hbn/clips.py`** -- Clip metadata (names, BIDS task names, file paths, trim points).

4. **Config TOML** for combined training (NMED + DS005876 + HBN).
