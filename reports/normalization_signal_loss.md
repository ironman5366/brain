# Per-Epoch Normalization Destroys EEG-Audio Signal

## Summary

All deep learning models trained on continuous EEG-audio data produced chance-level results. A diagnostic investigation using Temporal Response Functions (TRF) revealed that the data preprocessing pipeline independently Z-scores each 1-second EEG window, destroying the temporal amplitude dynamics required for auditory envelope tracking. Bypassing the pipeline and loading from the original researcher-provided data recovers statistically significant signal in 14 out of 20 subjects.

## The Problem

Every model architecture tested on the 32-channel continuous EEG-audio task failed:

| Model | Top-1 Retrieval | Song Classification | Cosine Similarity |
|-------|----------------|--------------------|--------------------|
| NMED-only baseline | ~0% | ~10% (chance) | 0.97 (collapsed) |
| SongFam-only baseline | ~0% | ~10% (chance) | 0.97 (collapsed) |
| Combined baseline | ~0% | ~10% (chance) | 0.97 (collapsed) |
| Hierarchical (w2-full) | ~0% | ~10% (chance) | ~0 (collapsed) |

The combined baseline showed embedding collapse (cosine similarity = 0.97 between all pairs), while hierarchical models collapsed to near-zero embeddings. Neither architecture could extract any auditory information from the EEG.

## Investigation: Is There Signal At All?

Before investing in more complex models, we needed to answer a fundamental question: **does this EEG data contain decodable auditory information?**

We implemented a Temporal Response Function (TRF) analysis — the gold-standard linear method in auditory neuroscience for testing EEG-audio relationships. The approach:

1. **Backward TRF (stimulus reconstruction)**: Ridge regression from time-lagged multichannel EEG to audio amplitude envelope
2. **Leave-one-song-out cross-validation**: 10-fold CV within each subject
3. **Permutation significance testing**: 200 circular-shift permutations per subject
4. **Audio features**: Broadband amplitude envelope via Hilbert transform, low-passed at 30 Hz, downsampled to 125 Hz

### First Run: Processed Data (32 channels, per-epoch normalized)

Using the data from our pipeline (`continuous-32ch/full/` safetensors):

| Metric | Value |
|--------|-------|
| Grand mean r | **0.003 +/- 0.002** |
| Significant subjects (p<0.05) | **1/20** |
| Null mean r | 0.0001 |

Only one subject (S20, r=0.019, p=0.03) reached significance — likely a false positive given 20 comparisons. This appeared to confirm that the data contained no signal.

### But These Are Published Datasets

The NMED-T dataset has been used in multiple peer-reviewed studies. Published work on music EEG shows reliable auditory envelope tracking (r ~ 0.01-0.05 for music, weaker than speech but consistently above chance). Something in our pipeline had to be wrong.

## Root Cause: Per-Epoch Z-Scoring

The data build pipeline (`data/nmed/build.py`, line 36) sets:

```python
NORMALIZATION = "epoch"
```

This propagates through `utils.py:standardize_epochs()` and `utils.py:map_egi_to_32ch()`, where each 1-second window (125 samples at 125 Hz) is independently Z-scored:

```python
# utils.py, lines 197-201
elif normalization == "epoch":
    x = standardized[:, mask, :]
    mean = x.mean(dim=-1, keepdim=True)   # mean over time within this window
    std = x.std(dim=-1, keepdim=True)     # std over time within this window
    standardized[:, mask, :] = (x - mean) / (std + eps)
```

**Why this destroys the signal:**

Auditory envelope tracking relies on the EEG amplitude covarying with the audio amplitude over time. When a loud passage plays, EEG amplitudes should be larger; during quiet passages, smaller. Per-epoch normalization forces every 1-second window to have mean=0 and std=1 *regardless of the actual audio level*, erasing exactly this relationship.

When these independently-normalized windows are concatenated into "continuous" sequences for the hierarchical models, the result is an artificial signal with standardized statistics per window and discontinuities at window boundaries — not the smooth temporal dynamics the models need.

## The Fix: Load From Original Data

The NMED-T dataset provides imputed `.mat` files that are already preprocessed by the original researchers (Chebyshev lowpass at 50 Hz, artifact rejection, NaN imputation, downsampled to 125 Hz). These are properly continuous recordings without per-epoch normalization.

### Second Run: Raw Imputed Data (125 channels, PCA to 64, no per-epoch normalization)

Loading directly from `songXX_Imputed.mat` files with PCA dimensionality reduction (125 -> 64 components, following Zuk et al. 2021):

| Metric | Value |
|--------|-------|
| Grand mean r | **0.015 +/- 0.002** |
| Significant subjects (p<0.05) | **14/20** |
| Null mean r | -0.0000 |

### Side-by-Side Comparison

| | Processed (epoch-norm) | Raw imputed (PCA 64) |
|--|----------------------|---------------------|
| Grand mean r | 0.003 | **0.015** (5x higher) |
| Significant subjects | 1/20 | **14/20** |
| Non-significant | 19/20 | 6/20 |

All per-subject results from the raw data run:

| Subject | r | p-value | Significant |
|---------|------|---------|-------------|
| S02 | 0.019 | 0.015 | Yes |
| S03 | 0.019 | 0.005 | Yes |
| S04 | 0.008 | 0.184 | No |
| S05 | 0.020 | 0.010 | Yes |
| S06 | 0.028 | 0.005 | Yes |
| S07 | 0.004 | 0.284 | No |
| S08 | -0.008 | 0.896 | No |
| S09 | 0.031 | 0.005 | Yes |
| S10 | 0.014 | 0.025 | Yes |
| S11 | 0.012 | 0.065 | Marginal |
| S12 | 0.022 | 0.005 | Yes |
| S13 | 0.018 | 0.010 | Yes |
| S14 | 0.008 | 0.085 | Marginal |
| S15 | 0.009 | 0.090 | Marginal |
| S16 | 0.015 | 0.005 | Yes |
| S17 | 0.015 | 0.020 | Yes |
| S19 | 0.013 | 0.030 | Yes |
| S20 | 0.019 | 0.010 | Yes |
| S21 | 0.013 | 0.035 | Yes |
| S23 | 0.020 | 0.005 | Yes |

The grand mean r = 0.015 is at the lower end of published music envelope tracking results, which is expected — Zuk et al. (2021) showed music reconstruction is significantly weaker than speech (where r ~ 0.05-0.15), especially below 1 Hz.

## Additional Validation

**Alpha (regularization) insensitivity**: Grid search over alpha = 1e2 to 1e7 showed stable results (r = 0.0154-0.0160), confirming the signal is robust and not an artifact of regularization tuning.

**PCA improves results slightly**: 125ch raw (r=0.0135) vs 64-component PCA (r=0.0149). PCA removes noise dimensions while retaining 92-99% of variance per subject.

## Implications

1. **The EEG data contains real auditory signal.** The per-epoch normalization in the build pipeline was destroying it before the models ever saw the data.

2. **Deep learning models should be retrained on properly normalized data.** Options:
   - Set `NORMALIZATION = "none"` and rebuild — let the model handle normalization
   - Set `NORMALIZATION = "recording"` — normalize across the entire recording, preserving temporal dynamics
   - Load directly from the imputed `.mat` files, bypassing the safetensors pipeline

3. **The signal is small but real (r ~ 0.015).** This sets a ceiling on what linear methods can extract. Deep learning may improve on this by capturing nonlinear relationships, but expectations should be calibrated — this is music (not speech), and the effect is inherently weaker.

## Reproducing These Results

```bash
# Run TRF analysis on raw imputed data with PCA
uv run python analysis/trf_analysis.py --pca-components 64

# View results interactively
uv run streamlit run viz/trf_results.py
```

## References

- Zuk, N.J., Murphy, J.W., Reilly, R.B., & Lalor, E.C. (2021). Envelope reconstruction of speech and music highlights stronger tracking of speech at low frequencies. *PLOS Computational Biology*.
- Losorelli, S., Nguyen, D.T., Dmochowski, J.P., & Kaneshiro, B. (2017). NMED-T: A Tempo-Focused Dataset of Cortical and Behavioral Responses to Naturalistic Music. *ISMIR*.
