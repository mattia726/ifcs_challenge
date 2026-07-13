# Hyperparameter Tuning Summary

This tuning pass searched compact, deterministic grids around the original
hand-selected boosting settings. The goal was not to run a huge black-box
optimization, but to check whether nearby values for tree size, learning rate,
regularization, row sampling, and column sampling improved the existing
out-of-fold pipeline.

## Search setup

The script is:

```text
tune_hyperparameters.py
```

It reuses the same feature engineering and model-specific categorical handling
as `challenge_pipeline.py`:

```text
ExtraTrees, LightGBM, XGBoost: one-hot encoding
HistGradientBoosting: integer-coded categoricals
CatBoost: native categorical features
```

The tuning script writes:

```text
outputs/hyperparameter_tuning_results.csv
outputs/hyperparameter_tuning_best_by_model.csv
outputs/hyperparameter_tuning_best_rank_ensemble.csv
outputs/hyperparameter_tuning_model_mix_comparison.csv
outputs_tuned/baseline_vs_tuned_cv_metrics.csv
outputs_tuned/oof_old_tuned_model_mix_comparison.csv
```

The first tuning pass used 3-fold CV with average precision as the primary
ranking metric. Then the best candidates were validated with the same 5-fold CV
used by the main pipeline.

## mtry / column sampling

For the one-hot models, the preprocessing creates 529 post-encoding features:

```text
44 numeric or engineered features
108 Province dummy columns
186 sector dummy columns
191 Ateco dummy columns
```

LightGBM and XGBoost both keep:

```text
colsample_bytree = 0.8
```

So their effective mtry-style value is:

```text
0.8 * 529 = 423 features per tree
```

CatBoost's closest equivalent is `rsm`. The best average-precision CatBoost
candidate kept `rsm = 1.0`, while the best CatBoost candidate by individual F1
used `rsm = 0.75`, equal to about 35 of the 47 raw features. The final promoted
pipeline keeps `rsm = 1.0` because the ensemble comparison favored the stronger
regularized CatBoost configuration.

## Promoted model settings

The final promoted configuration is a hybrid selected for the class-label
submission criterion:

```text
LightGBM: tuned
XGBoost: original setting kept
HistGradientBoosting: tuned
CatBoost: tuned
```

The original XGBoost setting was kept because the 5-fold OOF recombination
check showed it worked better inside the ensemble than the faster tuned XGBoost
candidate.

There is one tradeoff: the AP-only best blend keeps the original LightGBM as
well as the original XGBoost and reaches average precision 0.523771, but its
best-F1 is only 0.517505. Because the pipeline's default submission is based on
the best-F1 threshold, the promoted setup keeps tuned LightGBM and reaches the
better best-F1 of 0.520854.

### LightGBM

```text
n_estimators = 500
learning_rate = 0.035
num_leaves = 12
min_child_samples = 35
subsample = 0.85
colsample_bytree = 0.8
reg_alpha = 0.1
reg_lambda = 2.0
scale_pos_weight = negative_count / positive_count
```

### XGBoost

```text
n_estimators = 350
learning_rate = 0.04
max_depth = 3
min_child_weight = 4
subsample = 0.85
colsample_bytree = 0.8
reg_alpha = 0.05
reg_lambda = 2.0
scale_pos_weight = negative_count / positive_count
tree_method = hist
```

### HistGradientBoosting

```text
max_iter = 350
learning_rate = 0.035
l2_regularization = 0.1
max_leaf_nodes = 11
min_samples_leaf = 15
class_weight = balanced
```

### CatBoost

```text
iterations = 550
learning_rate = 0.035
depth = 5
l2_leaf_reg = 12
random_strength = 2.0
loss_function = Logloss
eval_metric = AUC
auto_class_weights = Balanced
```

## 5-fold before/after comparison

The comparison below uses the original `outputs/cv_metrics.csv` as the baseline
and the tuned run in `outputs_tuned/cv_metrics.csv` as the final validation.

| Model | Old ROC-AUC | Tuned ROC-AUC | Old AP | Tuned AP | Old best F1 | Tuned best F1 |
|---|---:|---:|---:|---:|---:|---:|
| ExtraTrees | 0.867091 | 0.867091 | 0.497182 | 0.497182 | 0.500319 | 0.500319 |
| LightGBM | 0.872316 | 0.874619 | 0.504978 | 0.506936 | 0.510751 | 0.510766 |
| XGBoost | 0.878262 | 0.878262 | 0.511832 | 0.511832 | 0.516598 | 0.516598 |
| HistGradientBoosting | 0.877521 | 0.879205 | 0.494314 | 0.501177 | 0.510775 | 0.516211 |
| CatBoost | 0.876765 | 0.878319 | 0.509279 | 0.511743 | 0.510596 | 0.510823 |
| Rank ensemble | 0.880242 | 0.880735 | 0.521607 | 0.523156 | 0.517105 | 0.520854 |

The gain is modest but consistent at the ensemble level:

```text
ROC-AUC:           +0.000494
Average precision: +0.001549
Best F1:           +0.003749
```

The best-F1 threshold remains:

```text
0.88
```
