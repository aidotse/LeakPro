import json

"""
Modify the json files so as to be used easier with the data_handling script
We collapse all annotators decisions, and consider them all as equally correct training examples
 """

import os

print("where: ", os.getcwd())

#os.chdir("examples/gia/pii_inverting_1_image")  # Change to where the script expects to be run from
#print("where: ", os.getcwd())
#print(os.path.exists("examples"))

#print(os.path.exists("examples/gia/pii_inverting_1_image"))


with open('echr_train.json', "r", encoding="utf-8") as f1, open('echr_dev.json', "r", encoding="utf-8") as f2, open('echr_test.json', "r", encoding="utf-8") as f3:

    train = json.load(f1)
    dev = json.load(f2)
    test = json.load(f3)

    training_raw = []
    dev_raw = []
    test_raw = []

    for ann_data in train:
        dct = {}
        dct['split'] = ann_data['dataset_type']
        dct['text'] = ann_data['text']
        dct['doc_id'] = ann_data['doc_id']
        did = ann_data['doc_id']
        for annotator in ann_data['annotations']:
            annotations = []
            for annotation in ann_data['annotations'][annotator]['entity_mentions']:
                if annotation['identifier_type'] != 'NO_MASK':
                    annotation['label'] = 'MASK'
                else:
                    annotation['label'] = 'NO_MASK'
                annotation['id'] = did
                annotation['span_text'] = annotation['span_text']
                annotations.append(annotation)

            dct['annotations'] = annotations
            training_raw.append(dct)

    for ann_data in dev:
        dct = {}
        dct['split'] = ann_data['dataset_type']
        dct['text'] = ann_data['text']
        dct['doc_id'] = ann_data['doc_id']
        did = ann_data['doc_id']
        for annotator in ann_data['annotations']:
            annotations = []
            for annotation in ann_data['annotations'][annotator]['entity_mentions']:
                if annotation['identifier_type'] != 'NO_MASK':
                    annotation['label'] = 'MASK'
                else:
                    annotation['label'] = 'NO_MASK'
                annotation['id'] = did
                annotation['span_text'] = annotation['span_text']
                annotations.append(annotation)

            dct['annotations'] = annotations
            dev_raw.append(dct)

    for ann_data in test:
        dct = {}
        dct['split'] = ann_data['dataset_type']
        dct['text'] = ann_data['text']
        dct['doc_id'] = ann_data['doc_id']
        did = ann_data['doc_id']
        for annotator in ann_data['annotations']:
            annotations = []
            for annotation in ann_data['annotations'][annotator]['entity_mentions']:
                if annotation['identifier_type'] != 'NO_MASK':
                    annotation['label'] = 'MASK'
                else:
                    annotation['label'] = 'NO_MASK'
                annotation['id'] = did
                annotation['span_text'] = annotation['span_text']
                annotations.append(annotation)

            dct['annotations'] = annotations
            test_raw.append(dct)





