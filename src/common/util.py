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
                if ner_id % 2 == 1:
                    ner_id -= 1
                new_ner_ids.append(ner_id)
            data[i]["ner_ids"] = new_ner_ids


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
    data, _ = get_all_data()

    print(data['train'][0].keys())
    print(data['valid'][0].keys())
    print(data['test'][0].keys())

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