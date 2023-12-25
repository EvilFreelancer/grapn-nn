import json
import os
import time
import logging
import requests

logger = logging.getLogger(__name__)


def get_page(page_num=0):
    # Reference for GET request parameters
    params = {
        "area": 1,  # The search is carried out by vacancies in the city of Moscow
        "page": page_num,  # Index of the search page on HH
        "per_page": 100,  # Amount vacancies on 1 page
        "period": 30,
        "label": "not_from_agency",
        "order_by": "publication_time",
    }
    req = requests.get("https://api.hh.ru/vacancies", params)

    # We decode his answer so that the Cyrillic alphabet is displayed correctly
    data = req.content.decode()
    req.close()
    return data


# Via public API possibly to get only 2000 vacancies
for page in range(0, 20):
    jsObj = json.loads(get_page(page))
    nextFileName = "./docs/pagination/{}.json".format(len(os.listdir("./docs/pagination")))
    f = open(nextFileName, mode="w", encoding="utf8")
    f.write(json.dumps(jsObj, ensure_ascii=False, sort_keys=True, indent=2))
    f.close()

    # Check to the last page if there are fewer than 2000 vacancies
    if (jsObj["pages"] - page) <= 1:
        break

logger.info("The search pages are collected")
