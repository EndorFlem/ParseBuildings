import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
import shap


categorical = [
    "region_num",
    "building_class",
    # "wall_material",
    # "finish_type_count",
    # "free_plan",
    # "developer_group",
    "address_code",
    # "energy_efficiency",
]
numerical = [
    "metro_time",
    "floors_max",
    "total_area",
    # "latitude",
    # "longitude",
    # "flats_count",
    # "parking_count",
    "inflation",
    "curs",
    "key_rate",
    # "price_for_sqm_adj",
]
text_features = [
    # "address",
    # "developer",
    "metro",
]
target_col = "price_for_sqm"

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")


all_data_df = pd.read_csv("data/new_buildings.csv")

for df in [train_df, val_df, test_df]:
    df["metro_time"] = df["metro_time"].fillna(-1)

for df in [train_df, val_df, test_df]:
    df["metro"] = df["metro"].fillna("None")

for df in [train_df, val_df, test_df]:
    df["finish_type_count"] = df["finish_type_count"].astype(int)

for col in categorical:
    df[col] = df[col].astype(str)

X_train = train_df[categorical + numerical + text_features]
y_train = np.log1p(train_df[target_col])

X_val = val_df[categorical + numerical + text_features]
y_val = np.log1p(val_df[target_col])

X_test = test_df[categorical + numerical + text_features]
y_test = test_df[target_col]


cat_features = [X_train.columns.get_loc(col) for col in categorical]

cat_features_idx = [X_train.columns.get_loc(c) for c in categorical]
text_features_idx = [X_train.columns.get_loc(c) for c in text_features]

model = CatBoostRegressor(
    iterations=1122,
    learning_rate=0.44617687687955854,
    depth=4,
    loss_function="RMSE",
    cat_features=cat_features,
    text_features=text_features,
    text_processing={
        "tokenizers": [
            {
                "tokenizer_id": "Space",
                "tokenizer_type": "Space",
            }
        ],
        "dictionaries": [
            {
                "dictionary_id": "Unigram",
                "dictionary_type": "Unigram",
            },
            {
                "dictionary_id": "BiGram",
                "dictionary_type": "Bigrams",
            },
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
    },
    early_stopping_rounds=100,
    verbose=100,
    use_best_model=True,
)

model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

test_df["predicted_price"] = y_pred
test_df["abs_error"] = (test_df[target_col] - y_pred).abs()

test_df["passed_test"] = (test_df["abs_error"] / test_df[target_col]) < 0.2

failed_tests_df = test_df[test_df["passed_test"] == False]
failed_tests_df.to_csv("failed_tests.csv", index=False)

mape = mean_absolute_percentage_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

mae_pct = mae / y_test.mean() * 100
rmse_pct = rmse / y_test.mean() * 100

print(f"MAE: {mae:.0f} руб/м² ({mae_pct:.2f}%)")
print(f"RMSE: {rmse:.0f} руб/м² ({rmse_pct:.2f}%)")

print(f"CatBoost MAPE на тесте: {mape * 100:.2f}%")


errors = np.abs(y_test - y_pred)

error_df = X_test.copy()
error_df["real_price"] = y_test
error_df["pred_price"] = y_pred
error_df["abs_error"] = errors
error_df["perc_error"] = errors / y_test * 100

worst_predictions = error_df.sort_values("abs_error", ascending=False)
print(worst_predictions.head(10))


test_pool = Pool(X_test, cat_features=cat_features, text_features=text_features)

explainer = shap.TreeExplainer(model)


shap_values = explainer.shap_values(test_pool)


shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X_test.columns)

shap.initjs()
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
    feature_names=X_test.columns,
)
