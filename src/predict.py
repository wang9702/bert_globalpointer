import os
from transformers import BertModel, BertTokenizerFast, BertConfig
from src.model import EffiGlobalPointer
import json
import torch
import numpy as np
from tqdm import tqdm
from src.config import ent2id, set_args


def NER_RELATION(text, ner_model, tokenizer,  args):
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=args.max_len, truncation=True)["offset_mapping"]
    new_span, entities= [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])

    encoder_txt = tokenizer.encode_plus(text, max_length=args.max_len)
    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(args.device)
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(args.device)
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(args.device)
    scores = ner_model(input_ids, attention_mask, token_type_ids)[0].data.cpu().numpy()
    scores[:, [0, -1]] -= np.inf
    scores[:, :, [0, -1]] -= np.inf
    for l, start, end in zip(*np.where(scores > 0)):
        entities.append({"start_idx":new_span[start][0], "end_idx":new_span[end][-1], "type":id2ent[l], "entity": text[new_span[start][0]:new_span[end][-1]+1]})
    return {"text":text, "entities":entities}


if __name__ == '__main__':

    args = set_args()
    model_path = f'{args.output_dir}/{args.task_name}/model.pth'
    id2ent = {}
    for k, v in ent2id.items():
        id2ent[v] = k

    tokenizer = BertTokenizerFast.from_pretrained(args.pre_model_dir)
    config = BertConfig.from_pretrained(args.pre_model_dir)
    encoder = BertModel(config)
    model = EffiGlobalPointer(encoder, len(list(ent2id.values())), 64).to(args.device)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()

    all_ = []
    test_path = os.path.join(args.data_dir, args.task_name, 'dev.json')
    for d in tqdm(json.load(open(test_path, 'r', encoding='utf8')), desc='Predict'):
        all_.append(NER_RELATION(d["text"], model, tokenizer, args))
    json.dump(
        all_,
        open(f'{args.output_dir}/{args.task_name}/test_result.json', 'w', encoding='utf8'),
        indent=4,
        ensure_ascii=False
    )

    # for d in json.load(open(r'D:\workspace\efficient_globalpointer\data\bill\bill_test.jsonl', 'r', encoding='utf8'))[:1]:
    #     time_start = time.time()
    #     res = NER_RELATION(d["text"], tokenizer= tokenizer, ner_model=model)
    #     time_end = time.time()
    #     print(time_end-time_start)
    #     print(res)

