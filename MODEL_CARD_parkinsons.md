# Model Card — Parkinson's Disease Classifier

**Version:** 1.0 · **Date:** 2026-06-07 · **License:** see repository

## Model details

- **Type:** Gradient Boosting Classifier (`sklearn.ensemble.GradientBoostingClassifier`)
- **Framework:** scikit-learn ≥ 1.8
- **Hyperparameters:** `learning_rate=0.1`, `max_depth=5`, `n_estimators=50`, `max_features="sqrt"`
- **Preprocessing:** `StandardScaler` on all features
- **Random seed:** 42 (fixed in [`configs/parkinsons.yaml`](configs/parkinsons.yaml))
- **Selected by:** Stage 2 grid search on Stage-1-best preprocessing; see
  [`reports/report.pdf`](reports/report.pdf) §4.2.6

## Intended use

Research benchmark comparing classical tabular classifiers on a
high-dimensional, mildly imbalanced binary problem.

**Out of scope:**
- Clinical decision support, patient triage, or any healthcare use
- Real-time or production deployment for individual diagnosis
- Generalisation to populations outside the Istanbul University training distribution
- Use without external validation on a held-out clinical cohort

## Training data

- **Source:** UCI ML Repository,
  [Parkinson's Disease Classification Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification)
- **Origin:** Istanbul University, 188 PD patients + 64 healthy controls
- **Recording:** ~3 sustained-vowel /a/ phonations per patient
- **Features:** 754 acoustic — MFCCs and their deltas, Tunable Q-Factor
  Wavelet Transform (TQWT), wavelet, vocal-fold, and baseline jitter/shimmer
  measurements
- **Total records:** 756
- **Class balance:** ~75% PD / ~25% Healthy (mildly imbalanced)

## Evaluation

- **CV protocol:** 10-fold `StratifiedGroupKFold` on the training set
  (recordings from the same patient never cross train/validation folds)
- **Held-out test split:** 20% patient-grouped split (no patient appears in
  both train and test); `random_state = 42`
- **Test set:** 153 recordings (111 PD + 42 Healthy)

## Performance

Held-out test set metrics (see [`artifacts/final/parkinsons/final_metrics.json`](artifacts/final/parkinsons/final_metrics.json)):

| Metric | Value |
|---|---|
| Accuracy | 0.850 |
| F1 (PD, primary) | 0.905 |
| Precision (PD) | 0.833 |
| Recall (PD) | 0.991 |
| Specificity (Healthy recall) | 0.476 |
| ROC-AUC | 0.938 |

Per-class breakdown:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Healthy | 0.952 | 0.476 | 0.635 | 42 |
| PD | 0.833 | 0.991 | 0.905 | 111 |

## Quantitative analysis

The model is **strongly biased toward predicting PD**: it correctly identifies
110 of 111 PD test cases (recall 99%) but mislabels 22 of 42 healthy controls
as PD (52% false-positive rate on the healthy class). Aggregate F1 = 0.905
reflects this asymmetry — Healthy F1 alone is 0.635.

ROC-AUC = 0.938 indicates the underlying score distribution has substantial
class separability that the default 0.5 decision threshold does not fully
exploit; threshold tuning could trade PD recall for Healthy specificity.

## Ethical considerations

1. **Not suitable for clinical use.** Single training source, small sample
   (~250 patients), no external validation, no calibration analysis.
2. **PD-prediction bias.** In a clinical context this would translate to
   overdiagnosis. Mitigations not implemented in this version: class
   weighting at training time (`class_weight="balanced"`); post-hoc threshold
   tuning on the decision function output.
3. **No fairness analysis.** Patient age, gender, and disease severity stage
   were not analysed as factors in performance.
4. **Reproducibility.** Pipeline + random seeds are fixed; the model artefact
   can be regenerated exactly from this repo.

## Caveats and recommendations

- **Threshold tuning** on `predict_proba` to find an operating point
  appropriate for the use case is the most natural next step (the ROC-AUC of
  0.938 leaves substantial headroom).
- **Class weighting** via `GradientBoostingClassifier(class_weight=...)` was
  out of scope for the original study; would partially address the bias.
- **Probability calibration** (e.g. `CalibratedClassifierCV`) was not
  performed. ROC-AUC is the only threshold-independent metric reported.
- **External validation** on an independent clinical cohort is required
  before any non-research use.

## Reproducibility

```bash
make install && make download && make reproduce
```

Or via Docker:

```bash
docker build -t data-science . && \
  docker run --rm -v "$(pwd)/artifacts:/app/artifacts" data-science make reproduce
```

Regenerates the model and all metrics quoted above. Random seed 42 throughout.
