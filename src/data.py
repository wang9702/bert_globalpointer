import json
from seqeval.metrics.sequence_labeling import get_entities


def get_train_dev(mode):
    with open(f'data/raw/{mode}/sentences.txt', encoding='utf8') as f:
        lines = f.read().splitlines()
        texts = []
        for line in lines:
            texts.append(''.join(line.split()))

    with open(f'data/raw/{mode}/tags.txt') as f:
        lines = f.read().splitlines()
        true_entities = []
        for line in lines:
            true_entities.append(get_entities(line.split()))

    train_jsonl = []
    for text, entities in zip(texts, true_entities):
        item = {'text': text, 'entities': []}
        for entity in entities:
            item['entities'].append({'start_idx':entity[1], 'end_idx':entity[2], 'type':entity[0], 'entity':text[entity[1]:entity[2]+1]})
        train_jsonl.append(item)
    json.dump(train_jsonl, open(f'data/processed/report_analyze/{mode}.json', 'w', encoding='utf8'), ensure_ascii=False, indent=4)

get_train_dev('train')