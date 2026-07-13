# IFCS Challenge Presentation Results

This document summarizes the results of the analysis for a 5-minute
presentation. It focuses on the story to tell, the key numbers to show, and the
artifacts that support each claim.

Technical implementation details are documented separately in
`pipeline_methodology.md`.

## Executive Message

Italian SMEs in the dataset are not simply "healthy" or "distressed". They form
a financial health ladder:

```text
high-profit cash generators
healthy profitable firms
thin-margin vulnerable firms
debt-burdened break-even firms
loss-making cash-flow distressed firms
```

The same variables that make this segmentation interpretable also drive the
predictive model:

```text
profit per employee
sector and ATECO activity
cash-flow quality
tax and financial-expense pressure
geography
```

The strongest supervised model is a rank ensemble of tree-based tabular models.
It reaches:

```text
ROC-AUC:           0.880
Average precision: 0.522
Best-F1 threshold: 0.88
OOF F1:            0.517
OOF precision:     0.512
OOF recall:        0.522
OOF accuracy:      0.895
```

The interpretation and prediction results are consistent: distress is most
visible where profitability, operating cash flow, and debt-service capacity are
weak.

## Suggested 5-Minute Slide Structure

### Slide 1: Problem and Dataset

Message:

```text
We profile Italian SMEs by financial structure and predict distress from FY2023
financial statements.
```

Key facts:

```text
13,956 SMEs
1,505 distressed firms
10.78% distress rate
108 provinces
306 sectors
268 ATECO codes
```

Useful artifacts:

```text
outputs/dataset_summary.json
outputs/target_distribution.csv
```

Speaking point:

The class imbalance matters. A naive model can get high accuracy by predicting
mostly non-distress, so we evaluate F1, balanced accuracy, recall, precision,
ROC-AUC, and average precision.

### Slide 2: Clustering Method

Message:

```text
Clusters were built from financial behavior only, then interpreted by sector
and geography afterward.
```

Input families:

```text
size and scale: revenue, employees
profitability: net income, operating income, margins
cash generation: operating cash flow and OCF margin
debt pressure: financial expenses and Alert Index
tax burden: current taxes and tax ratios
financial state flags: negative income, negative OCF, no financial expenses
```

Important design choice:

```text
Province, sector, ATECO, and Financial distress were excluded from clustering.
```

Reason:

Clusters should represent financial profiles first. Sector and geography are
used later to explain where each profile appears.

Useful artifacts:

```text
outputs/cluster_k_selection.csv
outputs/cluster_profiles.csv
```

### Slide 3: Five Financial Profiles

Message:

```text
The k=5 solution is economically interpretable and separates firms along a
clear financial health gradient.
```

Cluster summary:

| Cluster | Firms | Distress Rate | Median Net Margin | Median OCF Margin | Interpretation |
|---|---:|---:|---:|---:|---|
| A Resilient high-profit cash generators | 2,085 | 0.43% | 15.6% | 19.7% | Strong profits, strong cash flow, very low financial-expense pressure |
| B Healthy profitable core SMEs | 6,043 | 2.75% | 4.5% | 7.4% | The stable middle: profitable, cash-generative, moderate scale |
| C Thin-margin vulnerable operators | 4,745 | 15.68% | 0.7% | 3.2% | Low margin, lower productivity, more sensitive to shocks |
| D Debt-burdened break-even firms | 174 | 31.61% | -1.8% | 1.5% | Near break-even operations but high financial-expense burden |
| E Loss-making cash-flow distressed firms | 909 | 58.42% | -10.2% | -6.1% | Losses and negative operating cash flow dominate |

Useful artifacts:

```text
outputs/cluster_profiles.csv
outputs/cluster_pca.png
outputs/cluster_umap.png
outputs/graph_data_cluster_pca.csv
outputs/graph_data_cluster_umap.csv
```

Speaking point:

Cluster E is not just low-profit. It is structurally different: median net
income, operating income, and operating cash flow are all negative. Cluster D is
smaller but interesting because operating results are near break-even while
financial expenses are high.

### Slide 4: Sector Interpretation

Message:

```text
Sector composition helps explain why some financial profiles are more exposed
than others.
```

Most overrepresented sectors by cluster:

Cluster A:

```text
Auxiliary insurance and pension fund activities
Other professional, scientific and technical activities
Human health services
Architectural and engineering consultancy
Testing and technical analysis
```

Cluster B:

```text
Wholesale trade
Wholesale of ICT equipment
Wholesale of specialized products
Wholesale of machinery and supplies
Wholesale of food, beverages, and tobacco
```

Cluster C:

```text
Retail food and beverage trade
Restaurants and mobile food service
Retail cultural and recreational goods
Bars without kitchen
```

Cluster D:

```text
Rubber products
Printing and related services
Beverage manufacturing
Textile weaving
Waste collection
```

Cluster E:

```text
Research and experimental development
Sports activities
Renting and operating real estate
Leather and luggage manufacturing
Advertising
```

Useful artifact:

```text
outputs/cluster_sector_lift.csv
```

Speaking point:

The sector story is intuitive but not simplistic. Restaurants and retail show
up in the thin-margin cluster, while some capital- or project-intensive
activities appear among the loss-making and debt-burdened profiles.

### Slide 5: Geography

Message:

```text
The North contains many firms in absolute terms, but high-risk financial
profiles are relatively more concentrated in parts of the Centre, Islands, and
South.
```

Largest regional representation:

| Region | Firms | Dataset Share | Distress Rate |
|---|---:|---:|---:|
| Lombardia | 3,641 | 26.09% | 10.79% |
| Veneto | 1,643 | 11.77% | 7.61% |
| Emilia-Romagna | 1,350 | 9.67% | 9.78% |
| Lazio | 1,188 | 8.51% | 14.98% |
| Piemonte | 983 | 7.04% | 10.89% |
| Toscana | 955 | 6.84% | 12.15% |
| Campania | 892 | 6.39% | 10.20% |

High-risk cluster share:

High-risk clusters are:

```text
C Thin-margin vulnerable operators
D Debt-burdened break-even firms
E Loss-making cash-flow distressed firms
```

Top regions by high-risk cluster share:

| Region | High-Risk Cluster Share | Distress Rate |
|---|---:|---:|
| Molise | 51.43% | 8.57% |
| Lazio | 50.17% | 14.98% |
| Basilicata | 50.00% | 5.56% |
| Sicilia | 48.01% | 12.37% |
| Sardegna | 47.96% | 12.22% |
| Toscana | 47.43% | 12.15% |
| Abruzzo | 47.39% | 14.06% |
| Marche | 47.06% | 8.82% |
| Puglia | 45.79% | 13.11% |
| Calabria | 45.65% | 11.41% |

Useful artifacts:

```text
outputs/region_summary.csv
outputs/region_high_risk_cluster_share.csv
outputs/region_high_risk_cluster_share.png
outputs/graph_data_region_high_risk_cluster_share.csv
```

Speaking point:

The geography chart should be presented as profile concentration, not as a
causal claim. The data shows where financially fragile profiles are more common;
it does not by itself prove that location causes distress.

### Slide 6: Predictive Model

Message:

```text
A rank ensemble gives robust risk ordering across several tabular models.
```

Models in the ensemble:

```text
ExtraTrees
LightGBM
XGBoost
HistGradientBoosting
CatBoost
```

Why an ensemble:

```text
Different models capture different nonlinearities and categorical effects.
Rank averaging reduces dependence on probability calibration.
```

OOF performance:

| Threshold Choice | Threshold | Accuracy | Balanced Accuracy | F1 | Precision | Recall | Predicted Positive Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Best F1 | 0.88 | 0.895 | 0.731 | 0.517 | 0.512 | 0.522 | 11.00% |
| Best MCC | 0.89 | 0.899 | 0.722 | 0.514 | 0.533 | 0.497 | 10.06% |
| Recall >= 0.80 | 0.71 | 0.782 | 0.793 | 0.444 | 0.306 | 0.807 | 28.42% |
| Best balanced accuracy | 0.70 | 0.774 | 0.795 | 0.439 | 0.300 | 0.821 | 29.55% |

Ranking metrics:

```text
ROC-AUC:           0.880
Average precision: 0.522
```

Useful artifacts:

```text
outputs/cv_metrics.csv
outputs/oof_predictions.csv
outputs/rank_ensemble_thresholds.json
outputs/ensemble_oof_true_predicted.png
outputs/graph_data_ensemble_oof_true_predicted.csv
outputs/ensemble_oof_confusion_best_f1.csv
```

Speaking point:

If the evaluation metric rewards F1 or MCC, the threshold around 0.88 to 0.89
is most defensible. If the challenge rewards recall or balanced accuracy, the
threshold should be lowered to around 0.70 to 0.71.

### Slide 7: What Drives Predicted Distress

Message:

```text
The model is not relying on a single variable. It combines profitability,
sector, cash-flow quality, tax pressure, debt pressure, and geography.
```

Top CatBoost importance features:

| Rank | Feature | Interpretation |
|---:|---|---|
| 1 | net_income_per_employee | Productivity-adjusted profitability |
| 2 | sector | Activity-specific risk profile |
| 3 | cashflow_net_income_gap | Difference between accounting profit and cash generation |
| 4 | taxes_operating_income_ratio | Tax burden relative to operating profit |
| 5 | Ateco | More granular activity code |
| 6 | financial_expense_abs_operating_income_ratio | Debt-service pressure relative to operating result |
| 7 | revenue_per_employee | Productivity and business model scale |
| 8 | Province | Geographic risk signal |
| 9 | Net income signed_log1p | Profitability level |
| 10 | operating_margin | Core operating efficiency |

Useful artifact:

```text
outputs/catboost_feature_importance.csv
```

Speaking point:

The model's top features align with the cluster story. Distress is not only
"small firm" risk. It is primarily low productivity, weak profit conversion,
cash-flow weakness, and industry/geographic exposure.

## Key Results to Say Out Loud

1. The dataset has a 10.78% distress rate, so class imbalance matters.
2. Five financial clusters give an interpretable health gradient.
3. The healthiest cluster has only 0.43% distress.
4. The loss-making cash-flow cluster has 58.42% distress.
5. Thin-margin firms form a large vulnerable middle: 4,745 firms with 15.68%
   distress.
6. The high-risk profiles are relatively more common in Lazio, several central
   and southern regions, and the islands.
7. The predictive rank ensemble reaches 0.880 ROC-AUC and 0.522 average
   precision.
8. The best-F1 threshold gives balanced precision and recall around 0.51 to
   0.52.
9. The model's most important features match the economic interpretation:
   profit per employee, sector, cash-flow quality, tax burden, ATECO, and debt
   pressure.

## Recommended Presentation Flow

Use the following order for a clear 5-minute talk:

1. Start with the target imbalance and business objective.
2. Explain that clustering uses only financial variables.
3. Show the k=5 cluster table.
4. Show UMAP or PCA as visual evidence of the financial profiles.
5. Show the regional high-risk cluster chart.
6. Show the ensemble OOF true-vs-predicted chart.
7. End with feature importance and the practical takeaway.

## Recommended Figures

Use these figures in the slide deck:

```text
outputs/cluster_umap.png
outputs/region_high_risk_cluster_share.png
outputs/ensemble_oof_true_predicted.png
```

Optional supporting figure:

```text
outputs/cluster_pca.png
```

Use UMAP for the main presentation because it separates the profiles more
clearly. Use PCA only if the audience asks for a linear projection.

## Recommended Tables

Use these CSVs to build presentation tables:

```text
outputs/cluster_profiles.csv
outputs/region_high_risk_cluster_share.csv
outputs/cv_metrics.csv
outputs/catboost_feature_importance.csv
```

If the slide deck needs exact chart data, use:

```text
outputs/graph_data_cluster_umap.csv
outputs/graph_data_region_high_risk_cluster_share.csv
outputs/graph_data_ensemble_oof_true_predicted.csv
```

## Final Takeaway

The strongest story is that the unsupervised and supervised tasks agree.

The clusters reveal a financial health gradient driven by margins, cash flow,
productivity, and financial-expense pressure. The classifier then uses the same
economic signals, enriched with sector, ATECO, and geography, to rank firms by
distress risk.

This makes the solution useful both as a prediction engine and as an
interpretable profile of Italian SME financial vulnerability.
