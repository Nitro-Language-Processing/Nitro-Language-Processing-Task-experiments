import json

tag_to_id = dict()

def get_data(filepath, change_ner_tags=False, change_ner_ids=False, first_n=None):
    """
    returns a list of dictionaries
    each dictionary has the keys:
    id - unseful
    ner_tags - list of ner string tags for each token
    ner_ids - mapping of the same string tags towards ints
    tokens - list of tokens from that document
    space_after - list of bools useful for concatenation of tokens in order to form the text
    reconstructed_document
    """
    global tag_to_id
    with open(filepath) as f:
        data = json.load(f)

    for i, datapoint in enumerate(data):
        ner_tags = datapoint["ner_tags"]
        ner_ids = datapoint["ner_ids"]

        for (ner_tag, ner_id) in zip(ner_tags, ner_ids):
            if ner_tag != "O":
                ner_tag = ner_tag[ner_tag.find("-") + 1:]
            if not change_ner_ids:
                if ner_tag not in tag_to_id.keys():
                    tag_to_id[ner_tag] = ner_id

        tokens = datapoint["tokens"]
        spaces_after = datapoint["space_after"]
        document = ""
        start_chars = []
        end_chars = []
        for (token, space_after) in zip(tokens, spaces_after):
            start_char = len(document)
            document += token
            if space_after:
                document += " "
            end_char = len(document)
            start_chars.append(start_char)
            end_chars.append(end_char)

        if change_ner_tags:
            new_ner_tags = []
            for ner_tag in ner_tags:
                if ner_tag != "O":
                    ner_tag = ner_tag[ner_tag.find("-") + 1:]
                new_ner_tags.append(ner_tag)
            data[i]["ner_tags"] = new_ner_tags

        if change_ner_ids:
            new_ner_ids = []

            for ner_id in ner_ids:
                if ner_id == 0:
                    pass
                elif ner_id % 2 == 0:
                    ner_id = ner_id // 2
                elif ner_id % 2 == 1:
                    ner_id = ner_id // 2 + 1
                new_ner_ids.append(ner_id)
            data[i]["ner_ids"] = new_ner_ids

            # new_ner_ids = ner_ids

            # {'O': 0, 'PERSON': 1, 'QUANTITY': 23, 'NUMERIC': 25, 'NAT_REL_POL': 9, 'GPE': 5, 'DATETIME': 17,
            # 'ORG': 3, 'PERIOD': 19, 'EVENT': 11, 'FACILITY': 29, 'ORDINAL': 27, 'LOC': 7, 'MONEY': 21, 'WORK_OF_ART': 15, 'LANGUAGE': 13}

            new_ner_tags = data[i]["ner_tags"]

            # print(new_ner_tags, new_ner_ids)
            for (new_ner_tag, new_ner_id) in zip(new_ner_tags, new_ner_ids):
                if new_ner_tag not in tag_to_id.keys():
                    tag_to_id[new_ner_tag] = new_ner_id

        data[i]["reconstructed_document"] = document
        data[i]["start_char"] = start_chars
        data[i]["end_char"] = end_chars

    if isinstance(first_n, int):
        data = data[:first_n]

    return data, tag_to_id

def get_all_data(change_ner_tags=False, change_ner_ids=False, first_n=None):
    global tag_to_id
    data = dict()
    filepaths = ["ronec/data/train.json", "ronec/data/valid.json", "ronec/data/test.json"]
    dataset_types = ["train", "valid", "test"]
    for filepath, dataset_type in zip(filepaths, dataset_types):
        data[dataset_type], tag_to_id = get_data(filepath=filepath, change_ner_tags=change_ner_tags, change_ner_ids=change_ner_ids, first_n=first_n)

    return data, tag_to_id

def main():
    data, tag_to_id = get_all_data(change_ner_tags=True, change_ner_ids=True)
    print(tag_to_id)
    possible_ner_tags = set()
    possible_ner_ids = set()

    for key in data.keys():
        for i in range(len(data[key])):
            del data[key][i]['start_char']
            del data[key][i]['end_char']
            del data[key][i]['reconstructed_document']
            del data[key][i]['id']
            for ner_id in data[key][i]['ner_ids']:
                possible_ner_ids.add(ner_id)
            for ner_tag in data[key][i]['ner_tags']:
                possible_ner_tags.add(ner_tag)

    #    print(data['train'][0])
    print(data['valid'][0].keys())
    print(data['test'][0].keys())
    print(len(data["train"]) + len(data["valid"])  + len(data["test"]))
    import random
    final_data = data["train"] + data["valid"] + data["test"]

    random.shuffle(final_data)

    print(len(final_data))
    # print(final_data[-1])
    with open('training_LUCI_26.json', 'w') as fout:
        json.dump(final_data, fout)
    print(final_data[0])
    print(final_data[-1])
    print(final_data[-1].keys())
    print(len(possible_ner_ids), possible_ner_ids)
    print(len(possible_ner_tags), possible_ner_tags)


if __name__=="__main__":
    main()


"""
{"id": 5260, 

"ner_tags": ["O", "B-EVENT", "I-EVENT", "O", "O", "O", "O", "B-ORG", "I-ORG", "O", "B-ORG", "I-ORG", 
"I-ORG", "B-NUMERIC", "O", "B-NUMERIC", "O"], 

"ner_ids": [0, 11, 12, 0, 0, 0, 0, 3, 4, 0, 3, 4, 4, 25, 0, 25, 0], 

"tokens": ["În", "Cupa", "României", "la", "volei", "feminin", ":", "Dinamo", "București", "-", "C.S.U.", 
"Politehnica-Tender", "Timișoara", "3", "-", "1", "."], 

"space_after": [true, true, true, true, true, false, true, true, true, true, true, true, true, false, false, false, false]}
"""