# -*- coding: utf-8 -*-
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from src.config import ent2id


class EntDataset(Dataset):
    def __init__(self, file_path, args, tokenizer, istrain=True):
        self.args = args
        self.data = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.istrain = istrain

    def load_data(self, path):
        D = []
        for d in json.load(open(path, 'r', encoding='utf8')):
            D.append([d['text']])
            for e in d['entities']:
                start, end, label = e['start_idx'], e['end_idx'], e['type']
                if start <= end:
                    D[-1].append((start, end, ent2id[label])) 
        return D

    def encoder(self, item):
        if self.istrain:
            text = item[0]
            token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=self.args.max_len, truncation=True)["offset_mapping"]
            start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            # 将raw_text的下标 与 token的start和end下标对应
            encoder_txt = self.tokenizer.encode_plus(text, max_length=self.args.max_len, truncation=True)
            input_ids = encoder_txt["input_ids"]
            token_type_ids = encoder_txt["token_type_ids"]
            attention_mask = encoder_txt["attention_mask"]

            return text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask
        else:
            #TODO 测试
            pass

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []
        batch_labels_bin = []
        for item in examples:
            raw_text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask = self.encoder(item)

            labels = np.zeros((len(ent2id), self.args.max_len, self.args.max_len))
            # labels_binary = np.zeros((2, max_len, max_len))
            for start, end, label in item[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[label, start, end] = 1

                    # for end_ in range(start, end):
                    #     labels_binary[1, start, end_] = 1
                    # labels_binary[1, start, end] = 1

            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])

        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()
        return raw_text_list, batch_inputids, batch_attentionmask, batch_segmentids, batch_labels#, batch_labels_bin

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item

