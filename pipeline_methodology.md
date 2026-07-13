# IFCS Challenge Pipeline Methodology

This document explains the preprocessing, clustering, model fitting, ensembling,
thresholding, and prediction-generation steps implemented in
`challenge_pipeline.py`.

The pipeline is designed for the IFCS 2026 Data Challenge, where the goal is to:

1. Build interpretable financial clusters for Italian SMEs.
2. Predict the binary target `Financial distress`.
3. Produce reproducible artifacts for analysis, slides, and final submission.

Run the full pipeline from the repository root:

```powershell
python challenge_pipeline.py --outdir outputs
```

When `test_features.csv` is available in the repository root, the same command
also creates `predictions.csv` and several threshold-specific candidate
submissions.

## 1. Input Data

The main input file is:

```text
train.csv
```

The training data contains one row per firm. The key identifier and target are:

```text
Company ID
Financial distress
```

The target is binary:

```text
TRUE  = firm is in financial distress
FALSE = firm is not in financial distress
```

The current training set has:

```text
13,956 rows
15 original columns
1,505 distressed firms
12,451 non-distressed firms
10.78% positive target rate
```

The expected future test file is:

```text
test_features.csv
```

It should contain the same feature columns as `train.csv`, excluding
`Financial distress`.

## 2. Target Handling

Target parsing is handled by `bool_target`.

The function accepts boolean, numeric, or string versions of the target. This
keeps the pipeline robust to small formatting differences such as:

```text
TRUE/FALSE
True/False
1/0
boolean dtype
```

Internally, the target is converted to:

```text
1 = distressed
0 = not distressed
```

The target is never used as a feature.

## 3. Supervised Feature Engineering

Supervised features are created in `make_model_features`.

The input frame is copied, and `Financial distress` is dropped if it is present.
The identifier `Company ID` is also removed from the feature matrix. It is used
only later when writing predictions.

### 3.1 Categorical Variables

The categorical variables used for supervised modeling are:

```text
Province
sector
Ateco
```

Although `Ateco` is stored as a number in the CSV, it is treated as a category,
not as a continuous numeric variable:

```python
x["Ateco"] = x["Ateco"].astype(str)
```

This is important because ATECO values are activity codes. A code such as `620`
is not economically "larger" than `310` in a linear sense.

### 3.2 Alert Index and EXCELLENT

The original `Alert Index` column is mixed type. Most rows are numeric, but some
rows contain:

```text
EXCELLENT
```

The pipeline does not pass this raw mixed column to the models. Instead it
creates three derived features:

```text
Alert Index numeric
Alert Index is excellent
Alert Index signed_log1p
```

The conversion logic is:

```python
alert_num = pd.to_numeric(x["Alert Index"], errors="coerce")
x["Alert Index numeric"] = alert_num
x["Alert Index is excellent"] = alert_num.isna().astype(int)
x["Alert Index signed_log1p"] = signed_log1p(alert_num.fillna(0))
x = x.drop(columns=["Alert Index"])
```

This means:

```text
Numeric Alert Index value -> kept as numeric
EXCELLENT                 -> missing in Alert Index numeric
EXCELLENT                 -> 1 in Alert Index is excellent
EXCELLENT                 -> 0 in Alert Index signed_log1p
```

During EDA we verified that numeric `Alert Index` is exactly:

```text
Operating cash flow / Total financial expenses
```

and `EXCELLENT` corresponds to firms with:

```text
Total financial expenses = 0
```

So the special flag preserves useful information without inventing an arbitrary
numeric value for `EXCELLENT`.

### 3.3 Ratio Features

The pipeline creates margin, productivity, debt-pressure, tax, and accounting
quality features.

Profitability and cash-flow margins:

```text
net_margin
operating_margin
ocf_margin
```

Debt and financing pressure:

```text
financial_expense_sales_ratio
financial_expense_abs_operating_income_ratio
tax_shield_financial_expense_ratio
financial_expenses_minus_tax_shield
```

Productivity and per-employee measures:

```text
revenue_per_employee
net_income_per_employee
operating_income_per_employee
ocf_per_employee
```

Tax ratios:

```text
taxes_operating_income_ratio
taxes_sales_ratio
```

Accounting and cash-flow gap measures:

```text
cashflow_net_income_gap
operating_net_income_gap
```

All divisions use safe denominators. Zero denominators are replaced with
missing values before division. Infinite values are then converted to missing
values.

### 3.4 Negative and Zero Flags

The pipeline adds binary flags for financially meaningful thresholds:

```text
Net income <= 0
Operating Income <= 0
Operating cash flow <= 0
Current taxes <= 0
Sales Revenue <= 0
Total financial expenses <= 0
Tax shield <= 0
```

These flags matter because financial distress is strongly related to negative
profitability and negative cash-flow states.

### 3.5 Signed Log Features

Many financial variables are heavily skewed and can be negative. Ordinary log
transforms cannot handle negative values, so the pipeline uses a signed log
transform:

```python
signed_log1p(x) = sign(x) * log(1 + abs(x))
```

This preserves the sign while compressing large magnitudes.

Signed-log features are created for:

```text
Sales Revenue
Employees
Net income
Operating Income
Maximum deductible amount
Total financial expenses
Tax shield
Operating cash flow
Current taxes
```

### 3.6 Tax Shield Consistency Feature

EDA showed:

```text
Maximum deductible amount = 0.30 * Operating Income
Tax shield = min(max(0, Maximum deductible amount), max(0, Total financial expenses))
```

The pipeline therefore creates:

```text
tax_shield_cap_gap
```

This is the difference between observed `Tax shield` and the capped amount
implied by the formula. It is almost always zero, but it is retained as a
consistency signal.

## 4. Missing Values

The supervised models use two missing-value strategies depending on feature
type.

Numeric features:

```text
SimpleImputer(strategy="median")
```

Categorical features:

```text
SimpleImputer(strategy="constant", fill_value="MISSING")
```

This handles the one missing `sector` value in training and any missing
categorical values that might appear in the test data.

## 5. Dummy Variables and Categorical Encoding

For one-hot based models, categorical preprocessing is handled by
`build_preprocessor`.

The categorical columns are:

```text
Province
sector
Ateco
```

They are imputed with `"MISSING"` and then encoded with:

```python
OneHotEncoder(
    handle_unknown="ignore",
    min_frequency=5
)
```

The effects are:

```text
handle_unknown="ignore"
```

Unseen test categories do not crash the model. They receive all-zero values for
that feature group's known dummy columns.

```text
min_frequency=5
```

Rare categories are grouped into an infrequent-category bucket. This reduces
overfitting to tiny sectors or rare ATECO codes.

On the current training set, the one-hot categorical block creates:

```text
Province: 108 dummy columns
sector:   186 dummy columns
Ateco:    191 dummy columns
Total:    485 dummy columns
```

The full current supervised feature matrix has:

```text
44 numeric features
485 categorical dummy features
529 total transformed features for one-hot models
```

CatBoost is handled differently. It receives the categorical columns as native
categorical features rather than one-hot dummies.

HistGradientBoosting is also handled differently. It receives label-coded
categorical columns, which is less ideal but provides model diversity in the
ensemble.

## 6. Clustering Feature Engineering

Clustering features are created in `make_clustering_features`.

The clustering task is intentionally based on financial characteristics only.
The following variables are excluded from the clustering input:

```text
Financial distress
Company ID
Province
sector
Ateco
Region
Macroarea
```

The reason is interpretability. We want clusters that represent financial
profiles first, then we analyze how those profiles are distributed by geography
and industry afterward.

The clustering feature set includes:

Signed-log scale variables:

```text
Sales Revenue
Employees
Net income
Operating Income
Total financial expenses
Operating cash flow
Current taxes
```

Margins and ratios:

```text
net_margin
operating_margin
ocf_margin
financial_expense_sales_ratio
financial_expense_abs_op_income_ratio
taxes_sales_ratio
revenue_per_employee_log1p
net_income_per_employee_signed_log1p
ocf_per_employee_signed_log1p
alert_signed_log1p
```

Financial state flags:

```text
no_financial_expenses
negative_net_income
negative_operating_income
negative_ocf
```

Before clustering, continuous clustering features are winsorized:

```text
lower cap = 0.5th percentile
upper cap = 99.5th percentile
```

Then missing values are median-imputed and the matrix is robust-scaled:

```python
SimpleImputer(strategy="median")
RobustScaler()
```

Robust scaling is used because financial ratios and monetary variables contain
large outliers.

## 7. K-Means Clustering

Clustering is performed in `run_clustering`.

The pipeline fits KMeans for:

```text
k = 2, 3, 4, 5, 6, 7, 8
```

For each `k`, it records:

```text
minimum cluster size
maximum cluster size
silhouette score
Calinski-Harabasz score
Davies-Bouldin score
```

These diagnostics are saved to:

```text
outputs/cluster_k_selection.csv
```

The presentation-ready solution currently uses:

```text
k = 5
```

The five clusters are labeled economically as:

```text
A Resilient high-profit cash generators
B Healthy profitable core SMEs
C Thin-margin vulnerable operators
D Debt-burdened break-even firms
E Loss-making cash-flow distressed firms
```

The cluster assignments are saved to:

```text
outputs/cluster_assignments_train.csv
```

The cluster profile table is saved to:

```text
outputs/cluster_profiles.csv
```

## 8. Geography and Sector Profiling

After clustering, the pipeline adds region-level geography using an ISTAT
province-to-region mapping.

The mapping is cached locally as:

```text
outputs/province_region_map.csv
```

Two province naming issues are normalized:

```text
Reggio di Calabria -> Reggio Calabria
Lombardia          -> treated as Region Lombardia
```

The pipeline then produces:

```text
outputs/region_summary.csv
outputs/region_high_risk_cluster_share.csv
outputs/cluster_macroarea_mix.csv
outputs/cluster_sector_lift.csv
```

High-risk clusters are defined as:

```text
C Thin-margin vulnerable operators
D Debt-burdened break-even firms
E Loss-making cash-flow distressed firms
```

The regional chart uses:

```text
outputs/graph_data_region_high_risk_cluster_share.csv
outputs/region_high_risk_cluster_share.png
```

## 9. PCA and UMAP Cluster Visualizations

The PCA and UMAP visualizations are not used to fit the clusters. They are
two-dimensional projections of the same robust-scaled financial feature matrix
used by KMeans.

PCA outputs:

```text
outputs/cluster_pca_coordinates.csv
outputs/graph_data_cluster_pca.csv
outputs/cluster_pca.png
```

UMAP outputs:

```text
outputs/cluster_umap_coordinates.csv
outputs/graph_data_cluster_umap.csv
outputs/cluster_umap.png
```

The UMAP settings are:

```python
UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.10,
    metric="euclidean",
    random_state=42
)
```

UMAP is included to create a more locally faithful visualization of the
financial profiles. PCA remains useful as a linear baseline.

## 10. Supervised Models

The supervised task predicts:

```text
Financial distress
```

The pipeline uses a set of diverse tabular models. Each model is trained with
stratified cross-validation and contributes out-of-fold predictions.

### 10.1 ExtraTrees

ExtraTrees uses the one-hot preprocessor:

```text
numeric median imputation
categorical MISSING imputation
categorical one-hot encoding
```

Main settings:

```text
n_estimators = 450
class_weight = balanced_subsample
min_samples_leaf = 2
max_features = sqrt
```

In `--quick` mode, `n_estimators` is reduced to 150.

### 10.2 LightGBM

LightGBM also uses the one-hot preprocessor.

Main settings:

```text
n_estimators = 350
learning_rate = 0.04
num_leaves = 24
min_child_samples = 35
subsample = 0.85
colsample_bytree = 0.8
reg_alpha = 0.1
reg_lambda = 2.0
scale_pos_weight = negative_count / positive_count
```

In `--quick` mode, `n_estimators` is reduced to 120.

### 10.3 XGBoost

XGBoost uses the one-hot preprocessor.

Main settings:

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

In `--quick` mode, `n_estimators` is reduced to 120.

### 10.4 HistGradientBoosting

HistGradientBoosting uses the engineered feature set, but categorical variables
are converted to integer category codes:

```text
Province -> category code
sector   -> category code
Ateco    -> category code
```

Main settings:

```text
max_iter = 350
learning_rate = 0.035
l2_regularization = 0.1
max_leaf_nodes = 23
class_weight = balanced
```

In `--quick` mode:

```text
max_iter = 140
learning_rate = 0.04
```

### 10.5 CatBoost

CatBoost receives the categorical variables natively:

```text
Province
sector
Ateco
```

The categorical columns are filled with `"MISSING"` and converted to strings.
Their column indices are passed through `cat_features`.

Main settings:

```text
iterations = 550
learning_rate = 0.035
depth = 5
l2_leaf_reg = 8
loss_function = Logloss
eval_metric = AUC
auto_class_weights = Balanced
```

In `--quick` mode:

```text
iterations = 180
learning_rate = 0.04
```

CatBoost feature importance is exported to:

```text
outputs/catboost_feature_importance.csv
```

## 11. Cross-Validation

Cross-validation is handled by `run_cv_models`.

The default CV scheme is:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Stratification is important because the target is imbalanced.

For each model:

1. The model is fitted on the training folds.
2. It predicts probabilities on the validation fold.
3. These validation predictions are stored as out-of-fold predictions.
4. Metrics are computed from the full OOF vector.

OOF predictions are saved to:

```text
outputs/oof_predictions.csv
```

Metrics are saved to:

```text
outputs/cv_metrics.csv
```

## 12. Rank Ensemble

The final ensemble is a rank average of model OOF predictions.

For each model's OOF scores:

```python
rankdata(scores) / n_rows
```

Then the model ranks are averaged:

```python
rank_ensemble = mean(model_rank_scores)
```

The rank ensemble is used instead of a plain probability average because the
models have different probability calibration. Ranking focuses on ordering firms
by risk, which is often more stable across heterogeneous tree models.

The ensemble column is saved in:

```text
outputs/oof_predictions.csv
```

as:

```text
rank_ensemble
```

## 13. Metrics and Thresholding

The hidden challenge metric is not stated in the brief. The pipeline therefore
evaluates multiple threshold choices.

For each model and for the rank ensemble, the pipeline computes:

```text
ROC-AUC
average precision
accuracy
balanced accuracy
F1
precision
recall
Matthews correlation coefficient
predicted positive rate
```

Threshold candidates are:

```text
0.50
base_rate
best_f1
best_balanced_accuracy
best_mcc
recall_at_least_0.80
```

For the current full run, the rank-ensemble thresholds are saved to:

```text
outputs/rank_ensemble_thresholds.json
```

The current default submission threshold is:

```text
best_f1
```

At the current full-run threshold, the OOF diagnostic confusion table is saved
to:

```text
outputs/ensemble_oof_confusion_best_f1.csv
```

## 14. Ensemble OOF True-Vs-Predicted Plot

The pipeline writes a true-vs-predicted diagnostic for the rank ensemble:

```text
outputs/ensemble_oof_true_predicted.png
outputs/graph_data_ensemble_oof_true_predicted.csv
```

The plot shows:

```text
x-axis = true Financial distress class
y-axis = rank-ensemble OOF score
horizontal line = best-F1 threshold
diamond marker = median score by true class
X marker = mean score by true class
```

This plot is useful for explaining score separation. It shows whether true
distressed firms tend to receive higher model risk scores than non-distressed
firms.

## 15. Final Model Fitting

When `test_features.csv` is present, the pipeline fits each model on the full
training data:

```text
ExtraTrees
LightGBM
XGBoost
HistGradientBoosting
CatBoost
```

The same feature engineering and preprocessing logic is applied to the test
data.

For test predictions:

1. Each final model predicts a probability or risk score.
2. Scores are converted to ranks.
3. The rank scores are averaged.
4. The chosen threshold is applied.
5. Submission CSVs are written.

The full test probability file is:

```text
outputs/test_probabilities.csv
```

## 16. Submission Files

If `test_features.csv` exists, the pipeline writes multiple candidate
submission files:

```text
outputs/predictions_f1.csv
outputs/predictions_mcc.csv
outputs/predictions_balanced_accuracy.csv
outputs/predictions_recall_heavy.csv
outputs/predictions_base_rate.csv
```

It also writes the default challenge submission to:

```text
predictions.csv
```

The default file uses the F1-optimized threshold unless changed in the pipeline.

The submission schema is exactly:

```text
Company ID,pred_class
```

where:

```text
pred_class in {"TRUE", "FALSE"}
```

## 17. Main Output Artifacts

EDA:

```text
outputs/dataset_summary.json
outputs/target_distribution.csv
outputs/numeric_summary.csv
outputs/sector_risk.csv
outputs/province_risk.csv
outputs/flag_risk.csv
outputs/alert_index_report.json
```

Clustering:

```text
outputs/cluster_k_selection.csv
outputs/cluster_profiles.csv
outputs/cluster_assignments_train.csv
outputs/cluster_sector_lift.csv
outputs/cluster_macroarea_mix.csv
```

Geography:

```text
outputs/province_region_map.csv
outputs/region_summary.csv
outputs/region_high_risk_cluster_share.csv
outputs/graph_data_region_high_risk_cluster_share.csv
outputs/region_high_risk_cluster_share.png
```

Cluster visualizations:

```text
outputs/cluster_pca_coordinates.csv
outputs/graph_data_cluster_pca.csv
outputs/cluster_pca.png
outputs/cluster_umap_coordinates.csv
outputs/graph_data_cluster_umap.csv
outputs/cluster_umap.png
```

Supervised modeling:

```text
outputs/oof_predictions.csv
outputs/cv_metrics.csv
outputs/rank_ensemble_thresholds.json
outputs/catboost_feature_importance.csv
outputs/ensemble_oof_confusion_best_f1.csv
outputs/graph_data_ensemble_oof_true_predicted.csv
outputs/ensemble_oof_true_predicted.png
```

Test-time outputs, when `test_features.csv` exists:

```text
outputs/test_probabilities.csv
outputs/predictions_f1.csv
outputs/predictions_mcc.csv
outputs/predictions_balanced_accuracy.csv
outputs/predictions_recall_heavy.csv
outputs/predictions_base_rate.csv
predictions.csv
```

## 18. Reproducibility Notes

The pipeline uses fixed random seeds where practical:

```text
RANDOM_STATE = 42
```

This controls:

```text
StratifiedKFold shuffling
KMeans random states
PCA random state
UMAP random state
tree model random states
plot jitter
```

Some libraries may still show small differences across package versions or
hardware, especially UMAP, LightGBM, XGBoost, and CatBoost. The output artifacts
in `outputs/` capture the run used for the current analysis.

Required Python packages are listed in:

```text
requirements.txt
```

Install them with:

```powershell
python -m pip install -r requirements.txt
```
