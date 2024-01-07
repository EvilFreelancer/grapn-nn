# Импортируем необходимые библиотеки
import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
import os
import json
from bs4 import BeautifulSoup
from mapping.replacements import *
from datasets import Dataset, DatasetDict
import pyarrow as pa

# Загружаем модель и создаём экземпляр класса для извлечения навыков
nlp = spacy.load("ru_core_news_lg")
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

vacancies = []
for fl in os.listdir("./crawler/hh/docs/vacancies/"):
    if fl == ".gitignore":
        continue
    fileName = "./crawler/hh/docs/vacancies/{}".format(fl)
    f = open(fileName, encoding="utf8")
    json_obj = json.loads(f.read())
    f.close()
    if 'errors' in json_obj:
        continue
    job_description = json_obj['name'] + "\n" + BeautifulSoup(json_obj['description'], "lxml").text
    try:
        annotations = skill_extractor.annotate(job_description)
        if not annotations['results']['full_matches']:
            continue
        print(json_obj['id'])
        vacancies.append({
            'id': json_obj['id'],
            'name': json_obj['name'],
            'description': BeautifulSoup(json_obj['description'], "lxml").text,
            'annotations': annotations,
        })
    except:
        print("An exception occurred")

technologies_data = []
for vacancy in vacancies:
    technologies = set()

    full_matches = vacancy['annotations']['results']['full_matches']
    for match in full_matches:
        technology = match['doc_node_value'].lower()
        if technologies not in REMOVE:
            technologies.add(technology)

    ngram_scored = vacancy['annotations']['results']['ngram_scored']
    for ngram in ngram_scored:
        technology = ngram['doc_node_value'].lower()
        if technologies not in REMOVE:
            technologies.add(technology)

    for key, values in REPLACEMENTS.items():
        for value in values:
            if value in technologies:
                technologies.remove(value)
                technologies.add(key)

    # Adding the processed list of technologies to the list
    technologies_data.append(sorted(list(technologies)))

# Convert list of technologies to a PyArrow Array.
tech_array = pa.array(technologies_data)

# Generate table
columns = {
    'id': pa.array([vacancy['id'] for vacancy in vacancies]),
    'name': pa.array([vacancy['name'] for vacancy in vacancies]),
    'description': pa.array([vacancy['description'] for vacancy in vacancies]),
    'technologies': tech_array
}
table = pa.Table.from_pydict(columns)

dataset_dict = DatasetDict({'train': Dataset(table)})
dataset_dict.push_to_hub('evilfreelancer/headhunter')
