# 32-Channel Contrastive Model Evaluation Results

## Retrieval Metrics (EEG → Audio)

| Model | Window | N_val | Chance% | Top-1% | Top-5% | Top-10% | MRR | Med.Rank |
|-------|--------|------:|--------:|-------:|-------:|--------:|----:|---------:|
| nmed32-only | 1s | 5,708 | 0.018 | 0.04 | 0.12 | 0.12 | 0.0017 | 2855 |
| songfam-only | 1s | 1,786 | 0.056 | 0.22 | 0.62 | 0.90 | 0.0072 | 800 |
| combined | 1s | 7,494 | 0.013 | 0.07 | 0.15 | 0.28 | 0.0027 | 2009 |
| combined-nmed-val | 1s | 5,708 | 0.018 | 0.05 | 0.14 | 0.21 | 0.0020 | 2837 |
| hierarchical-w2 | w2 | 2,848 | 0.035 | 0.07 | 0.21 | 0.35 | 0.0032 | 1421 |
| hierarchical-w5 | w5 | 1,134 | 0.088 | 0.18 | 0.53 | 0.88 | 0.0073 | 567 |
| hierarchical-w10 | w10 | 562 | 0.178 | 0.36 | 1.07 | 1.78 | 0.0136 | 281 |
| hierarchical-w30 | w30 | 180 | 0.556 | 1.11 | 3.33 | 5.56 | 0.0359 | 89 |
| hierarchical-full | full | 20 | 5.000 | 10.00 | 30.00 | 50.00 | 0.2133 | 9 |

## Song Classification & Cosine Similarity

| Model | Window | Songs | Song Acc% | Chance% | Cos Mean | Cos Std |
|-------|--------|------:|----------:|--------:|---------:|--------:|
| nmed32-only | 1s | 10 | 10.35 | 10.0 | -0.0417 | 0.7341 |
| songfam-only | 1s | 111 | 1.18 | 0.9 | 0.2581 | 0.3546 |
| combined | 1s | 121 | 8.17 | 0.8 | 0.9535 | 0.1744 |
| combined-nmed-val | 1s | 10 | 10.51 | 10.0 | 0.9689 | 0.1341 |
| hierarchical-w2 | w2 | 10 | 10.67 | 10.0 | -0.0200 | 0.0001 |
| hierarchical-w5 | w5 | 10 | 9.70 | 10.0 | -0.0064 | 0.0002 |
| hierarchical-w10 | w10 | 10 | 9.61 | 10.0 | -0.0044 | 0.0001 |
| hierarchical-w30 | w30 | 10 | 10.00 | 10.0 | -0.0017 | 0.0002 |
| hierarchical-full | full | 10 | 10.00 | 10.0 | 0.0029 | 0.0014 |

## Key Comparisons

- **Best baseline**: combined-nmed-val — Song acc 10.51%, Top-1 0.05%
- **Best hierarchical**: hierarchical-w2 — Song acc 10.67%, Top-1 0.07%

### Song Classification (10-way NMED) Comparison
- Baseline (nmed32-only): 10.35%
- hierarchical-w2: 10.67% (+0.32pp)
- hierarchical-w5: 9.70% (-0.65pp)
- hierarchical-w10: 9.61% (-0.75pp)
- hierarchical-w30: 10.00% (-0.35pp)
- hierarchical-full: 10.00% (-0.35pp)
