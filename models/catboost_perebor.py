import json
import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
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
    # "price_for_sqm_adj",
]
text_features = ["address", "developer", "metro"]
target_col = "price_for_sqm"
all_features = categorical + numerical + text_features


for df in [train_df, val_df, test_df]:
    df["metro_time"] = df["metro_time"].fillna(-1)
    df["metro"] = df["metro"].fillna("None")
    for col in categorical:
        df[col] = df[col].astype(str)

X_train = train_df[categorical + numerical + text_features]
y_train = np.log1p(train_df[target_col])

X_val = val_df[categorical + numerical + text_features]
y_val = np.log1p(val_df[target_col])

X_test = test_df[categorical + numerical + text_features]
y_test = test_df[target_col]

cat_features_idx = [X_train.columns.get_loc(c) for c in categorical]
text_features_idx = [X_train.columns.get_loc(c) for c in text_features]

X_full = train_df[categorical + numerical + text_features]
y_train = np.log1p(train_df[target_col])

X_val_full = val_df[categorical + numerical + text_features]
y_val = np.log1p(val_df[target_col])


def objective(trial: optuna.Trial):
    selected_features = []
    global all_features
    for feat in all_features:
        if trial.suggest_categorical(f"include_{feat}", [True, False]):
            selected_features.append(feat)

    if len(selected_features) == 0:
        return float("inf")

    X_train = X_full[selected_features]
    X_val = X_val_full[selected_features]

    cat_features = [f for f in categorical if f in selected_features]
    text_features_selected = [f for f in text_features if f in selected_features]

    params = {
        "iterations": trial.suggest_int("iterations", 50, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.7, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "loss_function": "RMSE",
        "cat_features": cat_features if cat_features else None,
        "text_features": text_features_selected if text_features_selected else None,
        "early_stopping_rounds": 100,
        "verbose": False,
        "use_best_model": True,
        "text_processing": (
            {
                "tokenizers": [{"tokenizer_id": "Space", "tokenizer_type": "Space"}],
                "dictionaries": [
                    {"dictionary_id": "Unigram", "dictionary_type": "Unigram"},
                    {"dictionary_id": "BiGram", "dictionary_type": "Bigrams"},
                ],
                "feature_processing": {
                    "default": [
                        {
                            "tokenizers_names": ["Space"],
                            "dictionaries_names": ["Unigram", "BiGram"],
                            "feature_calcers": ["BoW"],
                        }
                    ]
                },
            }
            if text_features_selected
            else None
        ),
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_val)

    rmse = root_mean_squared_error(y_true, y_pred)
    return rmse


best_trials = []

for i in range(5):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True, n_startup_trials=100, seed=42 * i
        ),
    )
    study.optimize(objective, n_trials=800)
    best_trials.append(study.best_trial)
    print("Лучшие параметры:", study.best_params)
    print("Лучшее значение RMSE:", study.best_value)

with open("best_trials_catboost.json", "w", encoding="utf-8") as file:
    json.dump(
        [
            {"params": t.params, "value": t.value, "number": t.number}
            for t in best_trials
        ],
        file,
        ensure_ascii=False,
        indent=4,
    )
