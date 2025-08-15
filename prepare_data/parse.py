# это парсер данных с сайта

import json
import time
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from undetected_chromedriver import Chrome, By
import pandas as pd
import sys

step = 20
# max_buildings = step * 2
max_buildings = 1000
max_step = 200
current_start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
capcha = True

additional_info = []


def get_building_info(id, driver: Chrome):
    driver.get(
        f"https://наш.дом.рф/%D1%81%D0%B5%D1%80%D0%B2%D0%B8%D1%81%D1%8B/%D0%BA%D0%B0%D1%82%D0%B0%D0%BB%D0%BE%D0%B3-%D0%BD%D0%BE%D0%B2%D0%BE%D1%81%D1%82%D1%80%D0%BE%D0%B5%D0%BA/%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82/{id}"
    )

    elements = driver.find_elements(By.ID, "__NEXT_DATA__")

    if len(elements) == 0:
        elements = driver.find_elements(By.ID, "__NEXT_DATA__")

    script_element = elements[0]
    script_element = driver.find_element(By.ID, "__NEXT_DATA__")
    script_content = script_element.get_attribute("innerHTML")
    # print(driver.current_url)
    next_data = json.loads(script_content)

    props = pd.json_normalize(next_data.get("props", {}))
    desired_keys = {
        "flatsCount",
        "buildingClass",
        "wallMaterial",
        "freePlan",
        "parkingCount",
        "ceilHeight",
        "finishTypeCount",
        "objEnergyEfficiency",
    }
    selected_columns = [
        col for col in props.columns if col.split(".")[-1] in desired_keys
    ]

    result = props[selected_columns].to_dict(orient="records")[0]

    result = {key.split(".")[-1]: value for key, value in result.items()}

    return result


def write_buildings(i, region):
    global max_buildings
    global capcha
    global additional_info
    with open(f"pages/{region}-{i}.json", "w") as file:
        url = (
            f"https://xn--80az8a.xn--d1aqf.xn--p1ai/%D1%81%D0%B5%D1%80%D0%B2%D0%B8%D1%81%D1%8B/api/kn/object?offset={i}&limit=20&sortField=obj_publ_dt&sortType=desc&residentialBuildings=1&objClass=1%3A2%3A3&place={region}&fromQuarter=2020-01-01&toQuarter=2025-12-31"
            "http/s://xn--80az8a.xn--d1aqf.xn--p1ai/"
        )

        driver.get(url)

        # на этот раз была капча и пришлось делать костыль
        if capcha:
            input("wait for captcha")
            capcha = False

        json_text = driver.execute_script("return document.body.innerText;")

        data = json.loads(json_text)
        max_buildings = data["data"]["total"]

        for obj in data["data"]["list"]:
            obj_id = obj.get("objId")
            if obj_id and obj.get("objPriceAVG"):
                building_info = get_building_info(obj_id, driver)
                additional_info.append({"objId": obj_id, **building_info})

        file.write(json_text)


def save_to_csv(filename="add_info/building_add_info.csv"):
    global additional_info

    df = pd.DataFrame(additional_info)

    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Saved {len(additional_info)} records to {filename}")


for region in [78]:
    driver = uc.Chrome()
    i = current_start
    while i < max_buildings:
        # print(max_buildings)
        write_buildings(i, region)
        time.sleep(2)
        i += 20

    # print(current_start, max_step)
    if max_buildings % step != 0:
        write_buildings(max_buildings - max_buildings % step, region)

    save_to_csv(f"add_info/building_add_info{region}.csv")

    driver.quit()
    capcha = False
