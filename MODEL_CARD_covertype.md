# Model Card — Forest Cover Type Classifier

**Version:** 1.0 · **Date:** 2026-06-07 · **License:** see repository

## Model details

- **Type:** Gradient Boosting Classifier (`sklearn.ensemble.GradientBoostingClassifier`)
- **Framework:** scikit-learn ≥ 1.8
- **Hyperparameters:** `learning_rate=0.1`, `max_depth=10`, `n_estimators=200`, `max_features="sqrt"`
- **Preprocessing:** raw features (no scaling)
- **Random seed:** 42 (fixed in [`configs/covertype.yaml`](configs/covertype.yaml))
- **Selected by:** Stage 2 grid search on Stage-1-best preprocessing; see
  [`reports/report.pdf`](reports/report.pdf) §4.3.5

## Intended use

Research benchmark for classical tabular classifiers on a balanced
multiclass cartographic problem.

**Out of scope:**
- Deployment on the natural (heavily imbalanced) class distribution — this
  model was trained on a balanced subsample
- Operational forestry decisions or ecosystem-management workflows
- Generalisation to ecosystems outside Roosevelt National Forest, Colorado
- Use without further evaluation on the unbalanced distribution

## Training data

- **Source:** UCI ML Repository, [Forest Covertype Dataset](https://archive.ics.uci.edu/ml/datasets/Covertype)
- **Origin:** US Forest Service, Roosevelt National Forest (Colorado)
- **Coverage:** 30 m × 30 m cartographic cells
- **Features:** 54 total — 10 continuous (elevation, aspect, slope,
  distances to hydrology/roadways/fire points, three hillshade indices)
  plus 44 one-hot binary indicators (4 wilderness areas + 40 soil types)
- **Total records (full UCI dataset):** 581,012, heavily class-imbalanced
- **Used for training:** balanced subsample of 7,000 records — 1,000 per
  class — sampled at `random_state = 42`

## Evaluation

- **CV protocol:** 10-fold `StratifiedKFold` on the training set
- **Held-out test split:** 20% stratified split (`random_state = 42`)
- **Test set:** 1,400 cells (200 per class)

## Performance

Held-out test set metrics (see [`artifacts/final/covertype/final_metrics.json`](artifacts/final/covertype/final_metrics.json)):

| Metric | Value |
|---|---|
| Accuracy | 0.827 |
| Macro F1 (primary) | 0.824 |
| Macro Precision | 0.824 |
| Macro Recall | 0.827 |

Per-class breakdown:

| Cover type | Precision | Recall | F1 |
|---|---|---|---|
| Spruce/Fir | 0.768 | 0.695 | 0.730 |
| Lodgepole Pine | 0.680 | 0.595 | **0.635** |
| Ponderosa Pine | 0.857 | 0.780 | 0.817 |
| Cottonwood/Willow | 0.925 | 0.985 | **0.954** |
| Aspen | 0.786 | 0.935 | 0.854 |
| Douglas-fir | 0.826 | 0.855 | 0.840 |
| Krummholz | 0.926 | 0.945 | **0.936** |

## Quantitative analysis

The dominant error mode is systematic confusion between **Lodgepole Pine**
and **Spruce/Fir** — these two species share overlapping cartographic
profiles in this dataset (similar elevation, distances, hillshade). The
model misclassifies ~30% of Lodgepole Pine as Spruce/Fir and vice versa.
This pair-confusion caps macro F1 around 0.82 regardless of further model
or hyperparameter changes within the cartographic feature space.

Cottonwood/Willow and Krummholz are the easiest classes (F1 > 0.93). Their
cartographic signatures are distinct from the other species.

## Ethical considerations

1. **Balanced-subsample training.** The model has not been evaluated on the
   natural imbalanced distribution; performance there is unknown and likely
   degrades on rare classes.
2. **Geographic specificity.** Trained only on Roosevelt National Forest data;
   cartographic patterns may not transfer to other ecosystems.
3. **Forestry decisions affect ecosystem management** — the model is not
   validated for any operational use and could not be used to inform real
   management decisions without substantial further validation.

## Caveats and recommendations

- The macro F1 ceiling of ~0.82 reflects **feature limitations**, not model
  capacity. Breaking the Lodgepole / Spruce-Fir confusion would require
  features outside the original cartographic set (remote sensing imagery,
  tree morphology, climate normals).
- Evaluation on the natural (imbalanced) distribution is a necessary next
  step before any deployment-readiness claim.
- All three top-tier ensembles (Gradient Boosting, Random Forest, XGBoost)
  cluster within 0.005 macro F1 — selection of Gradient Boosting reflects
  a CV margin much smaller than a single standard deviation.

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
