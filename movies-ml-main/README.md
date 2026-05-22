Global movies (1950–2026) — supervised learning (RQ1–RQ7)

 1. What this is

Regression project: predict `revenue_million` from movie metadata and production/marketing/ratings signals.

Dataset file: `global_movies_dataset_1950_2026.csv` (placed in this folder).

2. Dataset source

Kaggle: `https://www.kaggle.com/datasets/suhanigupta04/global-movies-dataset-19502026`

3. Modeling setup (mirrors the California project structure)

- One notebook per research question: `RQ01_notebook.ipynb` … `RQ07_notebook.ipynb`
- Outputs (tables + figures) are saved to `./outputs` (created automatically).

 Target

- `revenue_million` (continuous regression)

 Features used (high-level)

Numeric candidates (imputed + scaled where needed):
- `release_year`, `runtime_min`, `imdb_rating`, `votes`, `budget_million`, `marketing_budget_million`,
  `metascore`, `audience_score`, `award_nominations`, `award_wins`

Categorical candidates (imputed + one-hot encoded):
- `genre`, `subgenre`, `director`, `lead_actor`, `lead_actress`, `country`, `language`, `streaming_platform`

Binary candidates:
- `franchise_flag`

Leakage-avoidance:
- Excludes `roi_pct`, `top_100_prob`, `blockbuster_flag` from predictors by default (they are derived/near-derived from revenue or are labels).

 4. Metrics

Reported on held-out test set unless otherwise stated:
- MAE
- RMSE
- R²

RQ6 additionally reports 5-fold CV RMSE and a mild numeric noise perturbation test.

5. Run locally (Windows example)

From this folder:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then open and Run All in each notebook, or execute with nbconvert:

```
jupyter nbconvert --execute --ExecutePreprocessor.timeout=1800 --to notebook RQ01_notebook.ipynb RQ02_notebook.ipynb RQ03_notebook.ipynb RQ04_notebook.ipynb RQ05_notebook.ipynb RQ06_notebook.ipynb RQ07_notebook.ipynb
```

Outputs go to `outputs/`.

