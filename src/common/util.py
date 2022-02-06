import json

tag_to_id = dict()

def get_data(filepath, change_ner_tags=False):
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
            for (ner_tag, ner_id) in zip(ner_tags, ner_ids):
                if ner_tag != "O":
                    ner_tag = ner_tag[ner_tag.find("-") + 1:]
                new_ner_tags.append(ner_tag)
            data[i]["ner_tags"] = new_ner_tags

        data[i]["reconstructed_document"] = document
        data[i]["start_char"] = start_chars
        data[i]["end_char"] = end_chars

    return data, tag_to_id

def get_all_data(change_ner_tags=False):
    global tag_to_id
    data = dict()
    filepaths = ["ronec/data/train.json", "ronec/data/valid.json", "ronec/data/test.json"]
    dataset_types = ["train", "valid", "test"]
    for filepath, dataset_type in zip(filepaths, dataset_types):
        data[dataset_type], tag_to_id = get_data(filepath, change_ner_tags)

    return data, tag_to_id

def main():
    data, _ = get_all_data()

    print(data['train'][0].keys())
    print(data['valid'][0].keys())
    print(data['test'][0].keys())

if __name__=="__main__":
    main()