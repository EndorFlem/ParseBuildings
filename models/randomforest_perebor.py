import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
import optuna


train_df = pd.read_csv("data/train.csv")
valid_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")


categorical = [
    "region_num",
    "building_class",
    "wall_material",
    "finish_type_count",
    "free_plan",
    "developer_group",
    "address_code",
    "energy_efficiency",
]
numerical = [
    "metro_time",
    "floors_max",
    "total_area",
    "latitude",
    "longitude",
    "flats_count",
    "parking_count",
    "inflation",
    "curs",
    "key_rate",
    "price_for_sqm_adj",
]

text_features = ["address", "developer", "metro"]
target_col = "price_for_sqm"
# all_features = categorical + numerical + text_features

target_col = "price_for_sqm"


# target_col = "price_for_sqm"

to_keep = categorical + numerical + text_features

train_df = train_df.select_dtypes(include=[np.number])
valid_df = valid_df.select_dtypes(include=[np.number])
test_df = test_df.select_dtypes(include=[np.number])

feature_cols = [col for col in train_df.columns if col != target_col and col != "id"]


def remove_features(df: pd.DataFrame, to_keep):
    cols_to_keep = [
        col for col in df.columns if any((col.startswith(prefix)) for prefix in to_keep)
    ]
    # print(cols_to_keep)
    return df[cols_to_keep]


X_train = remove_features(train_df[feature_cols], to_keep)
X_valid = remove_features(valid_df[feature_cols], to_keep)
X_test = remove_features(test_df[feature_cols], to_keep)

y_train = np.log1p(train_df[target_col])
y_valid = np.log1p(valid_df[target_col])
y_test_actual = test_df[target_col]


def group_features_by_prefix(columns):
    grouped = defaultdict(list)
    for col in columns:
        base = re.sub(r"(_\d+)$", "", col)
        grouped[base].append(col)
    return grouped


grouped_features = group_features_by_prefix(X_train.columns)


base_feature_names = list(grouped_features.keys())


class FeatureSelectorByPrefix(BaseEstimator, TransformerMixin):
    def __init__(self, selected_bases):
        self.selected_bases = selected_bases

    def fit(self, X, y=None):
        self.cols_to_keep_ = [
            col
            for col in X.columns
            if any(col.startswith(base) for base in self.selected_bases)
        ]
        return self

    def transform(self, X):
        return X[self.cols_to_keep_]


def objective(trial):
    selected_bases = []
    for base in base_feature_names:
        if trial.suggest_categorical(f"use_{base}", [True, False]):
            selected_bases.append(base)

    if not selected_bases:
        return 1e6

    n_estimators = trial.suggest_int("n_estimators", 50, 2000, step=50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt"])
    random_state = trial.suggest_int("random_state", 0, 1000)

    pipe = Pipeline(
        [
            ("feature_selector", FeatureSelectorByPrefix(selected_bases)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    n_jobs=12,
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred_log = pipe.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    rmse = root_mean_squared_error(y_test_actual, y_pred)

    return rmse


best_trials = []
for i in range(5):
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42 * i)
    )
    study.optimize(objective, n_trials=2000)

    best_trials.append(study.best_trial)
    print("Лучшие параметры:", study.best_params)
    print("Лучшее значение RMSE:", study.best_value)

with open("best_trials_randomforest.json", "w", encoding="utf-8") as file:
    json.dump(
        [
            {"params": t.params, "value": t.value, "number": t.number}
            for t in best_trials
        ],
        file,
        ensure_ascii=False,
        indent=4,
    )
