#!/usr/bin/env python
# coding: utf-8

# # RQ3 — Preprocessing ablation (same model)
# 
# **Research question (RQ3).** How sensitive is a strong baseline model to different preprocessing choices (scaling, categorical handling, feature inclusion)?
# 
# **Task:** regression to predict `revenue_million`. **Outputs:** CSV table in `./outputs`.

# ## Methodology (this notebook)
# 
# - Fix the estimator (Random Forest) and vary preprocessing setups.
# - Train/test split (75/25, `random_state=42`).
# - Report **MAE**, **RMSE**, **R²** for each setup.

# In[ ]:


# Setup: paths, load data, modeling frame (Global movies — regression)
from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

IS_KAGGLE = os.path.exists("/kaggle/input")
INPUT_ROOT = Path("/kaggle/input") if IS_KAGGLE else Path(".")
OUT = Path("/kaggle/working") if IS_KAGGLE else Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)
RQ_PREFIX = "RQ03"
RANDOM_STATE = 42
TARGET = "revenue_million"


def find_raw_table_path() -> Path:
    preferred = ("global_movies_dataset_1950_2026.csv",)
    found: list[Path] = []
    if IS_KAGGLE:
        for root, _, files in os.walk(INPUT_ROOT):
            for fn in files:
                p = Path(root) / fn
                if p.suffix.lower() in {".csv"}:
                    found.append(p)
    else:
        for p in INPUT_ROOT.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".csv"}:
                found.append(p)
    for name in preferred:
        for p in found:
            if p.name.lower() == name.lower():
                return p
    if found:
        return found[0]
    raise FileNotFoundError(
        "No CSV found. Add the dataset via Kaggle Add Input or place global_movies_dataset_1950_2026.csv next to this notebook."
    )


def prepare_modeling_df(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    numeric_cols = [
        "release_year",
        "runtime_min",
        "imdb_rating",
        "votes",
        "budget_million",
        "marketing_budget_million",
        "metascore",
        "audience_score",
        "award_nominations",
        "award_wins",
    ]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "franchise_flag" in d.columns:
        d["franchise_flag"] = pd.to_numeric(d["franchise_flag"], errors="coerce")

    cat_cols = [
        "genre",
        "subgenre",
        # high-cardinality fields intentionally excluded for speed
        "country",
        "language",
        "streaming_platform",
    ]
    for c in cat_cols:
        if c in d.columns:
            d[c] = d[c].fillna("missing").astype(str)

    d[TARGET] = pd.to_numeric(d[TARGET], errors="coerce")
    d = d.dropna(subset=[TARGET])
    if len(d) > 30000:
        d = d.sample(30000, random_state=RANDOM_STATE)
    return d


def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


RAW_PATH = find_raw_table_path()
df = pd.read_csv(RAW_PATH)
MODEL_DF = prepare_modeling_df(df)

DEFAULT_FEATURE_COLS = [
    "release_year",
    "runtime_min",
    "imdb_rating",
    "votes",
    "budget_million",
    "marketing_budget_million",
    "metascore",
    "audience_score",
    "award_nominations",
    "award_wins",
    "franchise_flag",
    "genre",
    "subgenre",
    "country",
    "language",
    "streaming_platform",
]
LEAKY_OR_LABEL_COLS = {"roi_pct", "top_100_prob", "blockbuster_flag"}
FEATURE_COLS = [c for c in DEFAULT_FEATURE_COLS if c in MODEL_DF.columns and c not in LEAKY_OR_LABEL_COLS]

print("Loaded:", RAW_PATH)
print("Rows (modeling):", len(MODEL_DF), "Features:", len(FEATURE_COLS))


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = MODEL_DF[FEATURE_COLS]
y = MODEL_DF[TARGET]

# Reduce runtime for notebook execution
X = X.sample(min(len(X), 5000), random_state=RANDOM_STATE)
y = y.loc[X.index]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)

from pandas.api.types import is_numeric_dtype

cat_cols = [c for c in FEATURE_COLS if not is_numeric_dtype(MODEL_DF[c])]
num_cols = [c for c in FEATURE_COLS if is_numeric_dtype(MODEL_DF[c])]


def make_preprocessor(scale_numeric: bool):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    oh = OneHotEncoder(handle_unknown="ignore")

    cat_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("oh", oh)]
    )

    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    )


def fit_eval(prep, name: str):
    est = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_estimators=60,
        min_samples_leaf=8,
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", prep), ("model", est)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return {"Setup": name, **regression_metrics(y_test, pred)}


rows = []

# A: default (scaled numeric)
rows.append(fit_eval(make_preprocessor(scale_numeric=True), "A_scaled_num"))

# B: drop rating signals (imdb/metascore/audience)

# C: variant without high-cardinality categoricals
hi_card = [c for c in ["director", "lead_actor", "lead_actress", "title"] if c in FEATURE_COLS]
X3 = MODEL_DF[[c for c in FEATURE_COLS if c not in hi_card]]
y3 = MODEL_DF[TARGET]
X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.25, random_state=RANDOM_STATE
)
cat_cols3 = [c for c in X3.columns if not is_numeric_dtype(MODEL_DF[c])]
num_cols3 = [c for c in X3.columns if is_numeric_dtype(MODEL_DF[c])]
prepC = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            ),
            num_cols3,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols3,
        ),
    ]
)
pipeC = Pipeline(
    [
        ("prep", prepC),
        (
            "model",
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_estimators=60,
                min_samples_leaf=8,
                n_jobs=-1,
            ),
        ),
    ]
)
pipeC.fit(X3_train, y3_train)
rows.append({"Setup": "C_drop_hi_card_cats", **regression_metrics(y3_test, pipeC.predict(X3_test))})

# B: drop rating signals (imdb/metascore/audience)
drop_cols = [c for c in ["imdb_rating", "metascore", "audience_score"] if c in FEATURE_COLS]
X2 = MODEL_DF[[c for c in FEATURE_COLS if c not in drop_cols]]
y2 = MODEL_DF[TARGET]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.25, random_state=RANDOM_STATE
)
cat_cols2 = [c for c in X2.columns if not is_numeric_dtype(MODEL_DF[c])]
num_cols2 = [c for c in X2.columns if is_numeric_dtype(MODEL_DF[c])]
prepE = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            ),
            num_cols2,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "oh",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            ),
            cat_cols2,
        ),
    ]
)
pipeE = Pipeline(
    [
        ("prep", prepE),
        (
            "model",
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_estimators=60,
                min_samples_leaf=8,
                n_jobs=-1,
            ),
        ),
    ]
)
pipeE.fit(X2_train, y2_train)
rows.append(
    {"Setup": "B_drop_ratings", **regression_metrics(y2_test, pipeE.predict(X2_test))}
)

tbl = pd.DataFrame(rows).sort_values("RMSE")
tbl.to_csv(OUT / f"{RQ_PREFIX}_table_preprocessing_ablation.csv", index=False)
tbl

