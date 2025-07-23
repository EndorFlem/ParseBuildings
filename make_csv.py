import pandas as pd
import os
import json

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
}

all_data = []

for path in os.listdir("pages"):
    with open(f"pages/{path}", "r", encoding="utf-8") as f:
        data = json.load(f)

    df_with_price = pd.json_normalize(data["data"]["list"])
    if "objPriceAVG" not in df_with_price.columns:
        continue
    all_data.append(df_with_price[df_with_price["objPriceAVG"].notna()].copy())


combined_df = pd.concat(all_data, ignore_index=True)
result_df = combined_df[list(csv_columns.keys())].rename(columns=csv_columns)

developer_counts = result_df["developer"].value_counts()

developer_mapping = {}
current_id = 1
for developer, count in developer_counts.items():
    if count <= 1:
        developer_mapping[developer] = 0
    else:
        developer_mapping[developer] = current_id
        current_id += 1

result_df["developer_group"] = result_df["developer"].map(developer_mapping)

result_df["address"] = result_df["address"].astype(str).str.strip()

address_mapping = {
    address: idx for idx, address in enumerate(result_df["address"].unique(), start=1)
}

result_df["address_code"] = result_df["address"].map(address_mapping)

result_df["completion_date"] = pd.to_datetime(
    result_df["completion_date"], errors="coerce"
)

adjustment_coeffs = {
    2022: {"stp": 1 / 0.845947, "lenobl": 1 / 0.887080},
    2023: {"stp": 1 / 0.862955, "lenobl": 1 / 0.885914},
    2024: {"stp": 1 / 0.942061, "lenobl": 1 / 0.952069},
}


def adjust_price(row):
    price = row["price_for_sqm"]
    date = row["completion_date"]

    if pd.isna(date):
        return price

    year = date.year
    month = date.month
    region = "stp" if row["region_num"] == 78 else "lenobl"

    if year not in adjustment_coeffs:
        return price

    if year == 2024 and month > 6:
        return price

    coeff = adjustment_coeffs[year][region]
    return price * coeff


result_df["price_for_sqm"] = result_df.apply(adjust_price, axis=1)


result_df.to_csv("new_buildings.csv", index=False, encoding="utf-8-sig")


# df = pd.read_csv("analytics.csv")


# df_lo_general = df[(df["Класс жилья"] == "Общий") & (df["Количество комнат"] == "Все")]

# y2022 = [
#     "Февраль 2022",
#     "Март 2022",
#     "Апрель 2022",
#     "Май 2022",
#     "Июнь 2022",
#     "Июль 2022",
#     "Август 2022",
#     "Сентябрь 2022",
#     "Октябрь 2022",
#     "Ноябрь 2022",
#     "Декабрь 2022",
# ]
# y2023 = [
#     "Февраль 2023",
#     "Март 2023",
#     "Апрель 2023",
#     "Май 2023",
#     "Июнь 2023",
#     "Июль 2023",
#     "Август 2023",
#     "Сентябрь 2023",
#     "Октябрь 2023",
#     "Ноябрь 2023",
#     "Декабрь 2023",
# ]
# first_half_2024 = [
#     "Январь 2024",
#     "Февраль 2024",
#     "Март 2024",
#     "Апрель 2024",
#     "Май 2024",
#     "Июнь 2024",
# ]
# second_half_2024 = [
#     "Июль 2024",
#     "Август 2024",
#     "Сентябрь 2024",
#     "Октябрь 2024",
#     "Ноябрь 2024",
#     "Декабрь 2024",
# ]

# results = []

# for idx, row in df_lo_general.iterrows():
#     region = row["Регион"]

#     avg_first_half = row[first_half_2024].mean()
#     avg_second_half = row[second_half_2024].mean()
#     avg_y2022 = row[y2022].mean()
#     avg_y2023 = row[y2023].mean()

#     coeff_first_half = avg_first_half / avg_second_half
#     coeff_y2022 = avg_y2022 / avg_second_half
#     coeff_y2023 = avg_y2023 / avg_second_half
#     coeff_second_half = 1

#     results.append(
#         {
#             "Регион": region,
#             "year": "2022",
#             "k": coeff_y2022,
#         }
#     )
#     results.append(
#         {
#             "Регион": region,
#             "year": "2023",
#             "k": coeff_y2023,
#         }
#     )
#     results.append(
#         {
#             "Регион": region,
#             "year": "fh2024",
#             "k": coeff_first_half,
#         }
#     )
#     results.append(
#         {
#             "Регион": region,
#             "year": "sh2024",
#             "k": coeff_second_half,
#         }
#     )


# coeffs_2024 = pd.DataFrame(results)

# print(coeffs_2024)
