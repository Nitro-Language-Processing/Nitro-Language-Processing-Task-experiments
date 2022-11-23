import pandas as pd
import json

filepath = "small_data/data.json"

tag_to_id = {'O': 0, 'PERSON': 1, 'QUANTITY': 23, 'NUMERIC': 25, 'NAT_REL_POL': 9, 'GPE': 5, 'DATETIME': 17,
             'ORG': 3, 'PERIOD': 19, 'EVENT': 11, 'FACILITY': 29, 'ORDINAL': 27, 'LOC': 7, 'MONEY': 21, 'WORK_OF_ART': 15, 'LANGUAGE': 13}
print(tag_to_id)
tup_list = list(tag_to_id.items())
print(tup_list)
tup_list = sorted(tup_list, key=lambda x: x[1])
print(tup_list)
new_tag_to_id = dict()
for i, (tag, id) in enumerate(tup_list):
    new_tag_to_id[tag] = i
print(new_tag_to_id)

tag_to_id = {'O': 0, 'PERSON': 1, 'ORG': 2, 'GPE': 3, 'LOC': 4, 'NAT_REL_POL': 5, 'EVENT': 6, 'LANGUAGE': 7, 'WORK_OF_ART': 8,
             'DATETIME': 9, 'PERIOD': 10, 'MONEY': 11, 'QUANTITY': 12, 'NUMERIC': 13, 'ORDINAL': 14, 'FACILITY': 15}

with open(filepath, "r") as f:
    data = json.load(f)

print(data)

dict_data = {"tokens": [],
             "ner_label": []}
import random


random_solution = {"tokens": [], "ner_label": []}


for elem in data:
    print(elem)
    tokens = elem["tokens"]
    labels = elem["ner_tags"]
    ner_ids = elem["ner_ids"]
    for (token, label, ner_id) in zip(tokens, labels, ner_ids):
        # assert ner_id == tag_to_id[label]
        dict_data["ner_label"].append(ner_id) #tag_to_id[label])
        dict_data["tokens"].append(token)

        random_solution["ner_label"].append(random.randint(0, 15))  # tag_to_id[label])
        random_solution["tokens"].append(token)

solution = pd.DataFrame.from_dict(dict_data)

solution.to_csv("small_data/solution.csv")
random_solution = pd.DataFrame.from_dict(random_solution)

random_solution.to_csv("small_data/random_solution.csv")
