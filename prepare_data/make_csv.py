import numpy as np
import pandas as pd
import os
import json

from sentence_transformers import SentenceTransformer

csv_columns = {
    "objId": "id",
    "objCommercNm": "name",
    "developer.shortName": "developer",
    "developer.devInn": "developer_inn",
    "objPriceAVG": "price_for_sqm",
    "shortAddr": "address",
    "latitude": "latitude",
    "longitude": "longitude",
    "metro.name": "metro",
    "metro.time": "metro_time",
    "objFloorMin": "floors_min",
    "objFloorMax": "floors_max",
    "objElemLivingCnt": "apartments",
    "objSquareLiving": "total_area",
    "objReady100PercDt": "completion_date",
    "objPublDt": "start_date",
    "objGuarantyEscrowFlg": "escrow",
    "objProblemFlg": "problem_flag",
    "rpdNum": "rpd_number",
    "rpdRegionCd": "region_num",
    "shortAddr": "address",
    "flatsCount": "flats_count",
    "buildingClass": "building_class",
    "wallMaterial": "wall_material",
    "finishTypeCount": "finish_type_count",
    "freePlan": "free_plan",
    "parkingCount": "parking_count",
    "objEnergyEfficiency": "energy_efficiency",
}

all_data = []

for path in os.listdir("pages"):
    with open(f"pages/{path}", "r", encoding="utf-8") as f:

        print(path)
        data = json.load(f)

    df_with_price = pd.json_normalize(data["data"]["list"])
    if "objPriceAVG" not in df_with_price.columns:
        continue
    all_data.append(df_with_price[df_with_price["objPriceAVG"].notna()].copy())

result_df = pd.concat(all_data, ignore_index=True)

add_info_df = pd.DataFrame([])
for path in os.listdir("add_info"):
    readed_df = pd.read_csv(f"add_info/{path}")
    add_info_df = pd.concat([add_info_df, readed_df])

result_df = result_df.merge(right=add_info_df, on="objId", how="left")
result_df = result_df[list(csv_columns.keys())].rename(columns=csv_columns)

developer_counts = result_df["developer"].value_counts()
developer_mapping = {}

for developer, count in developer_counts.items():
    category = int(np.floor(np.log1p(count)))
    developer_mapping[developer] = category

result_df["developer_group"] = result_df["developer"].map(developer_mapping)

result_df["address"] = result_df["address"].astype(str).str.strip()

address_mapping = {
    address: idx for idx, address in enumerate(result_df["address"].unique(), start=1)
}

result_df["address_code"] = result_df["address"].map(address_mapping)

result_df["completion_date"] = pd.to_datetime(
    result_df["completion_date"], errors="coerce"
)

# Индексы цен
idx = pd.read_csv("data/PriceIndex.csv")


# result_df["year_month"] = (
#     result_df["completion_date"].dt.to_period("M").dt.to_timestamp()
# )
# result_df["year_month"] = result_df["completion_date"].dt.strftime("%Y %B")
result_df["start_date"] = pd.to_datetime(result_df["start_date"], errors="coerce")
result_df["year_month"] = result_df["start_date"].dt.strftime("%Y %B")


idx = idx.rename(
    columns={
        "Класс жилья": "building_class",
        "Количество комнат": "bedrooms",
    }
)
idx["region_num"] = idx["Регион"].apply(
    lambda reg: 78 if reg == "Город Санкт-Петербург" else 47
)
idx["building_class"] = idx["building_class"].replace({"Бизнес-Элитный": "Бизнес"})
idx = idx.drop(columns=["Регион", "bedrooms"])


month_map = {
    "Январь": 1,
    "Февраль": 2,
    "Март": 3,
    "Апрель": 4,
    "Май": 5,
    "Июнь": 6,
    "Июль": 7,
    "Август": 8,
    "Сентябрь": 9,
    "Октябрь": 10,
    "Ноябрь": 11,
    "Декабрь": 12,
}


def convert_date(s):
    month_name, year = s.split()
    return pd.Timestamp(year=int(year), month=month_map[month_name], day=1).strftime(
        "%Y %B"
    )


idx_long = idx.melt(
    id_vars=["region_num", "building_class"],
    var_name="year_month",
    value_name="price_index",
)

idx_long["year_month"] = idx_long["year_month"].apply(convert_date)

base_date = pd.Timestamp(2025, 6, 1).strftime("%Y %B")

base_idx_dict = (
    idx_long[idx_long["year_month"] == base_date]
    .set_index(["region_num", "building_class"])["price_index"]
    .to_dict()
)

idx_long["adjustment_coeff"] = idx_long.apply(
    lambda row: base_idx_dict.get((row["region_num"], row["building_class"]), 1)
    / row["price_index"],
    axis=1,
)

adj_coeff_dict = idx_long.set_index(["region_num", "building_class", "year_month"])[
    "adjustment_coeff"
].to_dict()

result_df["price_for_sqm_adj"] = result_df.apply(
    lambda row: row["price_for_sqm"]
    * adj_coeff_dict.get(
        (row["region_num"], row["building_class"], row["year_month"]), 1
    ),
    axis=1,
)

# инфляция
inflation_df = pd.read_csv("data/Inflation.csv")
inflation_df["Дата"] = pd.to_datetime(inflation_df["Дата"], format="%d.%m.%Y")
inflation_df["year_month"] = inflation_df["Дата"].dt.strftime("%Y %B")


inflation_df = inflation_df.drop(columns=["Дата", "Цель по инфляции"])

inflation_df = inflation_df.rename(
    columns={
        "Ключевая ставка, % годовых": "key_rate",
        "Инфляция, % г/г": "inflation",
    }
)

result_df = result_df.merge(inflation_df, how="left", on="year_month")

# курс доллара
dollar_df = pd.read_csv("data/DollarRate.csv")

dollar_df["data"] = pd.to_datetime(dollar_df["data"], format="%m/%d/%Y")
dollar_df["year_month"] = dollar_df["data"].dt.strftime("%Y %B")
dollar_df = dollar_df.drop(columns=["nominal", "data", "cdx"])
dollar_df = dollar_df.groupby("year_month")["curs"].mean().reset_index()

result_df = result_df.merge(dollar_df, how="left", on="year_month")


def to_month_year_str(df, col):
    df[col] = pd.to_datetime(df[col]).dt.strftime("%B %Y")
    return df


# убираем выбросы (не стал делать отедльный файл , пусть тут лежит)
# попробовать без убирания total_area
for col in ["price_for_sqm", "total_area"]:
    Q1 = result_df[col].quantile(0.25)
    Q3 = result_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    result_df = result_df[
        (result_df[col] >= lower_bound) & (result_df[col] <= upper_bound)
    ]


result_df.to_csv("data/new_buildings.csv", index=False, encoding="utf-8-sig")
