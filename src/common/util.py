import json

tag_to_id = dict()

def get_data(filepath):
    """
    returns a list of dictionaries
    each dictionary has the keys:
    id - unseful
    ner_tags - list of ner string tags for each token
    ner_ids - mapping of the same string tags towards ints
    tokens - list of tokens from that document
    space_after - list of bools useful for concatenation of tokens in order to form the text
    reconstructed_text
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
        for (token, space_after) in zip(tokens, spaces_after):
            document += token
            if space_after:
                document += " "
        data[i]["reconstruct_document"] = document

    return data


def main():
    global tag_to_id
    data = dict()
    filepaths = ["ronec/data/train.json", "ronec/data/valid.json", "ronec/data/test.json"]
    dataset_types = ["train", "valid", "test"]
    for filepath, dataset_type in zip(filepaths, dataset_types):
        data[dataset_type] = get_data(filepath)

    print(data['train'][0].keys())
    print(data['valid'][0].keys())
    print(data['test'][0].keys())

if __name__=="__main__":
    main()