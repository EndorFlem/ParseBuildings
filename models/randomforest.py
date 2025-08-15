import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

train_df = pd.read_csv("data/train.csv")
valid_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

categorical = [
    "region_num_",
    "building_class_emb",
    "wall_material_emb",
    # "finish_type_count",
    "free_plan",
    "developer_group",
    # "address_code",
    # "developer_inn",
]
numerical = [
    # "metro_time",
    "floors_max",
    # "total_area",
    # "latitude",
    # "longitude",
    # "flats_count",
    "parking_count",
    # "apartments",
]
text_features = [
    # "address",
    # "developer_emb",
    "metro",
]
target_col = "price_for_sqm"

to_keep = categorical + numerical + text_features

train_df = train_df.select_dtypes(include=[np.number])
valid_df = valid_df.select_dtypes(include=[np.number])
test_df = test_df.select_dtypes(include=[np.number])


feature_cols = [col for col in train_df.columns if col != target_col and col != "id"]


def remove_features(df: pd.DataFrame, to_keep):
    cols_to_keep = [
        col for col in df.columns if any((col.startswith(prefix)) for prefix in to_keep)
    ]
    return df[cols_to_keep]


X_train = remove_features(train_df[feature_cols], to_keep)
X_valid = remove_features(valid_df[feature_cols], to_keep)
X_test = remove_features(test_df[feature_cols], to_keep)

y_train = np.log1p(train_df[target_col])
y_valid = np.log1p(valid_df[target_col])
y_test_actual = test_df[target_col]

rf = RandomForestRegressor(
    n_estimators=10,
    max_depth=None,
    min_samples_split=9,
    min_samples_leaf=4,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    verbose=0,
)

rf.fit(X_train, y_train)

y_pred_log = rf.predict(X_test)
y_pred = np.expm1(y_pred_log)

mae = mean_absolute_error(y_test_actual, y_pred)
mape = mean_absolute_percentage_error(y_test_actual, y_pred)
rmse = root_mean_squared_error(y_test_actual, y_pred)

print(f"📊 MAE: {mae:.0f} руб/м")
print(f"📊 RMSE: {rmse:.0f} руб/м")
print(f"📉 MAPE: {mape * 100:.2f}% средняя ошибка")

print("\n📉 SHAP анализ фичей...")
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_valid)

shap_df = pd.DataFrame(np.abs(shap_values.values), columns=X_valid.columns)


def get_base_feature_name(col):
    return re.sub(r"_\d+$", "", col)


shap_df.columns = [get_base_feature_name(col) for col in shap_df.columns]
mean_shap_importance = shap_df.mean().groupby(level=0).sum().sort_values(ascending=True)

plt.figure(figsize=(10, 6))
mean_shap_importance.plot(kind="barh")
plt.title("Aggregated SHAP Importance by Original Feature")
plt.ylabel("Mean |SHAP value|")
plt.tight_layout()
plt.show()
