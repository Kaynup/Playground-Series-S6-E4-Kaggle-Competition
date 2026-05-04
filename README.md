# In-Fold Target Encoding Stacking Pipeline

**Filename:** `infold_te_stacking_pipeline.ipynb`  
**Task:** Multiclass classification — predicting irrigation need (Low / Medium / High) from agricultural sensor data.  
**Metric:** Balanced Accuracy (equal weight per class, regardless of class frequency).

---

## What This Pipeline Does

This is a two-level stacking ensemble. The first level trains three gradient-boosted tree models (XGBoost, LightGBM, CatBoost) across stratified k-folds. The second level trains a logistic regression on the out-of-fold probability outputs of those models, augmented with ensemble-level meta-features. Final predictions are post-processed via Nelder-Mead class weight optimization.

The defining technical property is that **all target encoding happens strictly within each training fold**. This prevents the encoded statistics from seeing validation labels, which would be a data leak that inflates OOF scores.

---

## Architecture Overview

```mermaid
flowchart TD
    A([Raw train.csv / test.csv]) --> B[Encode Labels\nLow=0  Medium=1  High=2]
    B --> C[Mass Feature Engineering\napplied jointly to train + test]

    C --> D1[Threshold Flags\nsoil_lt_25  wind_gt_10  etc]
    C --> D2[Domain Ratios\nmoist_rain  ET_proxy  heat_stress  etc]
    C --> D3[Formula Score\nrule-based High/Medium/Low vote]
    C --> D4[N-gram Categoricals\nBigrams + Trigrams over top cat cols]
    C --> D5[Binned x Categorical Interactions\nqcut bins x top_cat_cols]
    C --> D6[Group Aggregates\nmean diff ratio per cat group]
    C --> D7[Pairwise Factorized Features\nall pairs from top_cols]
    C --> D8[Rounding and Decimal Features\nrounded variants of key numerics]

    D1 & D2 & D3 & D4 & D5 & D6 & D7 & D8 --> E[Full Feature Matrix\nX_train shape N x ~200+ cols]

    style A fill:#1a1a2e,color:#e0e0e0,stroke:#4a4a8a
    style B fill:#16213e,color:#e0e0e0,stroke:#4a4a8a
    style C fill:#16213e,color:#e0e0e0,stroke:#4a4a8a
    style E fill:#0f3460,color:#e0e0e0,stroke:#4a4a8a
```

---

## Fold Loop: In-Fold Target Encoding

The key invariant: `TargetEncoder` is fit_transform-ed on the training split of each fold, then transform-ed on the validation split and test set. Validation labels never touch the encoder fit.

```mermaid
flowchart LR
    subgraph FOLD ["Fold k  (repeated N_FOLDS times)"]
        direction TB
        F1[Split X_tr / X_val] --> F2[TargetEncoder.fit_transform on X_tr]
        F2 --> F3[TargetEncoder.transform on X_val + X_test]
        F3 --> F4[Append TE columns\nTE_feature_class0/1/2]
        F4 --> F5[Drop high-cardinality raw cols\nPAIRS + NUM_CAT_AGG + BIN_CAT_INT]
        F5 --> F6[compute_sample_weight balanced\non y_tr]
    end

    F6 --> XGB[XGBoost\n3 seeds avg\nearly stop 150]
    F6 --> LGB[LightGBM\n3 seeds avg\nearly stop 150]
    F6 --> CB[CatBoost\n3 seeds avg\nearly stop 150]

    XGB --> OOF_X[oof_proba xgb va_idx]
    LGB --> OOF_L[oof_proba lgb va_idx]
    CB  --> OOF_C[oof_proba cb va_idx]

    style FOLD fill:#0d2137,color:#c9d6e3,stroke:#2d6a9f
    style XGB fill:#1b4332,color:#d8f3dc,stroke:#52b788
    style LGB fill:#1b3a4b,color:#caf0f8,stroke:#48cae4
    style CB fill:#3d1a3a,color:#f3d8f8,stroke:#c77dff
```

---

## Meta-Feature Construction

After the fold loop each model has produced full OOF probability arrays (shape N x 3) and test probability arrays averaged over folds. These are fed into `compute_meta_features`.

```mermaid
flowchart TD
    M1[oof_proba xgb — shape Nx3] --> STACK[np.stack shape 3xNx3]
    M2[oof_proba lgb — shape Nx3] --> STACK
    M3[oof_proba cb  — shape Nx3] --> STACK

    STACK --> MF1[mean_proba across models Nx3]
    STACK --> MF2[std_proba  across models Nx3]

    MF1 --> META1[mean_Low  mean_Medium  mean_High]
    MF2 --> META2[std_Low   std_Medium   std_High]
    MF2 --> META3[var_Low   var_Medium   var_High]
    MF1 --> META4[margin: top1 minus top2 probability]
    MF1 --> META5[max_prob: highest class probability]
    MF1 --> META6[entropy: neg sum p log p]
    STACK --> META7[agreement: fraction of models\nagreeing with mean argmax]

    META1 & META2 & META3 & META4 & META5 & META6 & META7 --> X_META[X_meta_oof\nhstack raw probas + meta feats]

    style STACK fill:#1a1a2e,color:#e0e0e0,stroke:#6a5acd
    style X_META fill:#0f3460,color:#e0e0e0,stroke:#6a5acd
```

---

## Level-2: Logistic Regression Stacking and Post-hoc Tuning

```mermaid
flowchart TD
    A[X_meta_oof\nshape N x 19] --> B[StratifiedKFold\nN_FOLDS=5  seed=SEED+999]
    B --> C[LogisticRegression\nC=1.0  max_iter=500\nbalanced sample weight]
    C --> D[stack_oof Nx3\nstack_test avg over folds]

    D --> E[Evaluate balanced_accuracy_score\nstack_oof argmax vs y_train]
    E --> F[Nelder-Mead Class Weight Tuning]

    F --> G[5 random seeds\nrandom half-splits of OOF\n2 directions each = 10 runs]
    G --> H[3 starting points per run\n1,1,1  /  1.2,0.9,1  /  1.4,0.85,1.05]
    H --> I[stack_w = mean of all found optima]

    I --> J[final_preds = stack_test times stack_w argmax]
    J --> K[submission_tuned_v2.csv]

    style A fill:#16213e,color:#e0e0e0,stroke:#4a4a8a
    style F fill:#2d1b3d,color:#e0e0e0,stroke:#9b59b6
    style I fill:#0f3460,color:#e0e0e0,stroke:#4a4a8a
    style K fill:#1b4332,color:#d8f3dc,stroke:#52b788
```

---

## Design Decisions and Why They Matter

### In-Fold Target Encoding (not global)

Target encoding computes per-category conditional probability statistics relative to the label. If you compute these on the full training set, the validation fold's labels participated in computing the encoding — every sample's encoded value is influenced by its own label. That is a leak.

The fix: fit the encoder only on `X_tr` (the in-fold training split), then apply it to `X_val`. The validation-set encoding is computed from statistics that never saw those labels.

`sklearn.preprocessing.TargetEncoder` with `target_type='multiclass'` produces three output columns per input feature (one per class), representing the conditional probability that a given category value belongs to each class.

### Shallow Trees with Early Stopping

All three models are capped at shallow depth (XGB `max_depth=3`, LGB `num_leaves=15`, CB `depth=4`) and use `n_estimators=2600` with early stopping at 150 rounds. Shallow trees reduce variance at the cost of some bias — appropriate when the feature space is wide and many features are correlated.

### Seed Averaging

Each model is trained `N_SEEDS=3` times per fold with seeds `SEED`, `SEED+1`, `SEED+2`. The per-seed probabilities are averaged before writing to `oof_proba`. This reduces noise from random subsampling and initialization without requiring extra CV folds.

### Balanced Sample Weighting

`compute_sample_weight('balanced', y_tr)` upweights minority class samples. This matters because the metric is balanced accuracy, which penalizes per-class errors equally regardless of class frequency.

### Nelder-Mead Weight Tuning

After stacking, `stack_oof` (shape N×3) is multiplied element-wise by a 3-vector `weights`, then argmax is taken. Nelder-Mead optimizes those 3 weights to maximize balanced accuracy on half-splits of the OOF data. The half-split prevents the tuned weights from just memorizing the full OOF distribution.

The process runs over 5 random seeds and 2 split directions each (10 total optimization runs), with 3 starting points per run to avoid local minima. `stack_w` is the mean across all resulting weight vectors.

---

## Feature Engineering Summary

| Group | Description |
|---|---|
| Threshold flags | Binary indicators for domain cutoffs (soil < 25, rain < 300, etc.) |
| Domain ratios | Ratios encoding water balance physics (ET_proxy, drying_force, water_deficit, etc.) |
| Formula score | Rule-based vote for High/Medium/Low, encoded as `formula_pred` plus components |
| N-gram categoricals | Pairwise and triple combos of top cat cols, factorized to integer codes |
| Bin x cat interactions | Each top_num_col binned into 5 quantiles, crossed with each top_cat_col |
| Group aggregates | Mean, diff, and ratio of each top_num_col within each top_cat_col group |
| Rounding variants | Rounded values at different decimal places for key numerics |
| Q-bins | Each top_num_col quantile-binned into 10 bins |
| Pairwise factorized | All pairs from top_cols as joint category codes |
| In-fold TE | Multiclass TE expanding every pre-TE feature column into 3 probability columns |

---

## Reproducibility

`SEED = 42` is propagated to StratifiedKFold, TargetEncoder, all base model random states, and the meta-fold split. `seed_everything()` seeds numpy and sets `PYTHONHASHSEED` at pipeline start.

---

## Configuration

```python
SEED        = 42
N_FOLDS     = 5
N_SEEDS     = 3
DATA_DIR    = Path('/kaggle/input/competitions/playground-series-s6e4/')
WORKING_DIR = Path('/kaggle/working/')
```

Output: `submission_tuned_v2.csv` written to `WORKING_DIR`.
