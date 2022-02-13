import spacy
from src.common.util import *
import numpy as np
from sklearn.metrics import f1_score
nlp = spacy.load("ro_core_news_lg")
from tqdm import tqdm
from statistics import mean


def finetune_spacy_engine():
    pass

def main():
    data, tag_to_id = get_all_data(change_ner_tags=True, change_ner_ids=True)
    train = data["train"]
    valid = data["valid"]
    test = data["test"]
    spacy_labels = set()
    dataset_labels = set(list(tag_to_id.keys()))

    scores = []
    spacy_to_dataset_labels = {"NUMERIC_VALUE": "NUMERIC", "ORGANIZATION":  "ORG"}

    spacy_vecs = []

    for doc in tqdm(test):
        spacy_doc = nlp(doc["reconstructed_document"])
        # vec = []
        start_chars_gt = doc["start_char"]
        end_chars_gt = doc["end_char"]
        cur_gt_labels = list(zip(doc["ner_tags"], start_chars_gt, end_chars_gt))
        cur_pred_labels = []
        for ent in spacy_doc.ents:
            if ent.label_ == "PRODUCT":
                continue
            spacy_label = ent.label_
            if ent.label in ["ORGANIZATION", "NUMERIC_VALUE"]:
                spacy_label = spacy_to_dataset_labels[spacy_label]

            cur_pred_labels.append((spacy_label, ent.start_char, ent.end_char))

        def vectorize(cur_labels):
            new_cur_labels = []
            # print(cur_labels)
            for i in range(len(cur_labels)):
                if cur_labels[i][0] == 'O':
                    continue
                label = cur_labels[i][0]
                if label in ["ORGANIZATION", "NUMERIC_VALUE"]:
                    label = spacy_to_dataset_labels[label]
                new_cur_labels.append((tag_to_id[label], cur_labels[i][1], cur_labels[i][2]))
            return new_cur_labels

        cur_gt_labels = vectorize(cur_gt_labels)
        cur_pred_labels = vectorize(cur_pred_labels)
        # print(len(cur_gt_labels), len(cur_pred_labels))
        # print(cur_pred_labels)
        # print(cur_gt_labels)

        cur_pred_dict = dict()
        cur_gt_dict = dict()

        for cur_gt_label in cur_gt_labels:
            s, e = cur_gt_label[1], cur_gt_label[2]
            lbl = cur_gt_label[0]
            for i in range(s, e+1):
                cur_gt_dict[i] = lbl

        for cur_pred_label in cur_pred_labels:
            s, e = cur_pred_label[1], cur_pred_label[2]
            lbl = cur_pred_label[0]
            for i in range(s, e + 1):
                cur_pred_dict[i] = lbl


        # print(len(cur_gt_dict), len(cur_pred_dict))


        for key in cur_gt_dict.keys():
            if key not in cur_pred_dict.keys():
                if key >= len(doc["reconstructed_document"]):
                    continue
                if doc["reconstructed_document"][key] != ' ':
                    cur_pred_dict[key] = tag_to_id["O"]

        for key in cur_pred_dict.keys():
            if key not in cur_gt_dict.keys():
                if key >= len(doc["reconstructed_document"]):
                    continue
                if doc["reconstructed_document"][key] != ' ':
                    cur_gt_dict[key] = tag_to_id["O"]

        common_keys = set(list(cur_pred_dict.keys())).intersection(set(list(cur_gt_dict.keys())))
        cur_gt_dict = {common_key: cur_gt_dict[common_key] for common_key in common_keys}
        cur_pred_dict = {common_key: cur_pred_dict[common_key] for common_key in common_keys}

        cur_gt_dict = sorted(cur_gt_dict.items())
        cur_pred_dict = sorted(cur_pred_dict.items())

        # print(cur_gt_dict, len(cur_gt_dict))
        # print()
        # print(cur_pred_dict, len(cur_pred_dict))

        final_gt_labels = [elem[1] for elem in cur_gt_dict]
        final_pred_labels = [elem[1] for elem in cur_pred_dict]


        # print(len(final_pred_labels), len(final_gt_labels), len(doc["reconstructed_document"]))

        scores.append(f1_score(final_pred_labels, final_gt_labels, average="weighted"))
        # print(scores[-1])

    print(mean(scores))
        # print(cur_gt_dict)
        # print(cur_pred_dict)
        # assert len(cur_gt_dict) == len(cur_pred_dict)

            # print(ent.text, ent.label_)
            # print(ent.start_char, ent.end_char, ent.text, doc["reconstructed_document"][ent.start_char:ent.end_char])

    #         spacy_labels.add(ent.label_)
    #         vec.append([ent.text, ent.label_])
    #     spacy_vecs.append(vec)
    # np.save(file="src/lucian/spacy_vec.npy", arr=np.array(spacy_vecs), allow_pickle=True)
    # print(len(dataset_labels), len(spacy_labels))
    # print(len(dataset_labels.intersection(spacy_labels)))
    # print(len(dataset_labels.union(spacy_labels)))
    # print(len(dataset_labels.intersection(spacy_labels)) / len(dataset_labels.union(spacy_labels)))
    #
    # print(dataset_labels - spacy_labels)
    # # {'O', 'NUMERIC', 'ORG'}, in
    # print(spacy_labels - dataset_labels)
    # # {'PRODUCT', 'ORGANIZATION', 'NUMERIC_VALUE'}


if __name__=='__main__':
    main()