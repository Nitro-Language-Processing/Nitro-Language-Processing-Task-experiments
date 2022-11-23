from src.common.util import get_all_data
from collections import Counter

data, tag_to_id = get_all_data(change_ner_tags=True, change_ner_ids=True)

train = data["train"]
valid = data["valid"]
test = data["test"]

print(type(train))
all = train + valid + test
cntr = Counter()

dic = dict()
for elem in all:
    elem_keys = elem.keys()
    tokens = elem['tokens']
    ner_tags = elem['ner_tags']
    # print(ner_tags)
    # print(tokens)
    # exit()
    last_ner_tag = ""
    new_word = ""
    for (token, ner_tag) in zip(tokens, ner_tags):
        if ner_tag == "O":
            last_ner_tag = ""
            new_word = ""
            continue

        # if ner_tag == last_ner_tag:
        #     new_word += (token + " ")

        if ner_tag != "O":
            last_ner_tag = ner_tag
            new_word += (token + " ")

        if ner_tag not in dic.keys():
            dic[ner_tag] = [new_word]
        else:
            dic[ner_tag].append(new_word)

print(len(dic))

for (key, val) in dic.items():
    print("*" * 30)
    print(f"Ner tag type: {key}, most common 50 examples: ", Counter(val).most_common(50))


with open("src/common/common_ner_examples.txt", "w+") as f:
    for (key, val) in dic.items():
        f.write("*" * 30)
        f.write("\n")
        entities = [val[0] for val in Counter(val).most_common(50)]
        f.write(f"Ner tag type: {key}:")
        f.write("\n")
        for ent in entities:
            f.write(f"     -> {ent}")
            f.write("\n")
        f.write("\n")