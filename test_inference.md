# Test Set Inference

The repository is prepared to generate challenge test predictions from saved
full-data models.

## Saved model bundle

The final full-data models are saved in:

```text
outputs/models/
```

The bundle contains:

```text
extra_trees.joblib
lightgbm.joblib
xgboost.joblib
hgb_labelcoded.joblib
catboost_native.joblib
model_manifest.csv
model_metadata.json
```

`model_metadata.json` stores the rank-ensemble thresholds from cross-validation,
including the default best-F1 threshold used for `predictions.csv`.

## Rebuild the model bundle

From the repository root:

```bash
python challenge_pipeline.py
```

The pipeline now fits and saves the final full-data models even when
`test_features.csv` is not present.

To save the bundle somewhere else:

```bash
python challenge_pipeline.py --model-dir outputs/models
```

## Run inference when the test set arrives

Place the challenge test file at:

```text
test_features.csv
```

Then run:

```bash
python predict_from_saved_models.py --test test_features.csv --model-dir outputs/models --outdir outputs
```

This writes:

```text
outputs/test_probabilities.csv
outputs/predictions_f1.csv
outputs/predictions_balanced_accuracy.csv
outputs/predictions_mcc.csv
outputs/predictions_recall_heavy.csv
outputs/predictions_base_rate.csv
predictions.csv
```

The root-level `predictions.csv` is copied from `outputs/predictions_f1.csv`,
matching the pipeline default.

## Current generated test predictions

The current repository state includes predictions generated from the provided
`test_features.csv` using the saved bundle in `outputs/models/`.

```text
predictions.csv
outputs/test_probabilities.csv
outputs/predictions_f1.csv
outputs/predictions_balanced_accuracy.csv
outputs/predictions_mcc.csv
outputs/predictions_recall_heavy.csv
outputs/predictions_base_rate.csv
```
