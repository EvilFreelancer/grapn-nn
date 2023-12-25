import json
import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

for fl in os.listdir("./docs/pagination/"):
    if fl == ".gitignore":
        continue
    pageName = "./docs/pagination/{}".format(fl)
    print(pageName)
    f = open(pageName, encoding="utf8")
    json_text = f.read()
    f.close()
    json_obj = json.loads(json_text)

    for v in json_obj["items"]:
        req = requests.get(v["url"])
        data = req.content.decode()
        req.close()

        logger.info(v["id"])
        fileName = "./docs/vacancies/{}.json".format(v["id"])
        print(fileName)
        f = open(fileName, mode="w", encoding="utf8")

        jsonVocObj = json.loads(data)
        f.write(json.dumps(jsonVocObj, ensure_ascii=False, sort_keys=True, indent=2))
        f.close()
