import random
import torch
import numpy as np
import json

BUFSIZE = 4096000


def ListsToTensor(xs, tknizer=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        if tknizer is not None:
            y = tknizer.token2idx(x) + [tknizer.padding_idx] * (max_len - len(x))
        else:
            y = x + [0] * (max_len - len(x))
        ys.append(y)
    return ys


def batchify(data, tknizer):
    truth, inp, msk = [], [], []
    for x in data:
        inp.append(x[:-1])
        truth.append(x[1:])
        msk.append([1 for i in range(len(x) - 1)])
    truth = torch.LongTensor(ListsToTensor(truth, tknizer)).t_().contiguous()
    inp = torch.LongTensor(ListsToTensor(inp, tknizer)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return truth, inp, msk


def s2t(strs, tknizer):
    inp, msk = [], []
    for x in strs:
        inp.append([w for w in x])
        msk.append([1 for i in range(len(x))])
    inp = torch.LongTensor(ListsToTensor(inp, tknizer)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return inp, msk


def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def parse_lines(lines, max_len, min_len):
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        l = json.loads(line)
        # for different dataset
        if 'text' in l:
            line = l['text'].strip()
        else:
            line = l['instruction'].strip()+l['output'].strip()
        if not line:
            continue
        line = [w for w in line]
        sents = chunks(line, max_len)
        if len(sents[-1]) < min_len:  # the last one is too short
            sents = sents[:-1]
        data.extend(sents)
    return data

def parse_lines_classification(lines, max_len, min_len):
    
    data = []
    for line in lines:
        # 以 '_!_' 分割每行数据
        parts = line.strip().split('_!_')
        news_id = parts[0]
        category_code = parts[1]
        category_name = parts[2]
        news_title = parts[3]
        keywords = parts[4].split(',')
        
        # 创建一个字典来存储每条新闻的信息
        news_item = {
            "news_id": news_id,
            "category_code": category_code,
            "category_name": category_name,
            "news_title": news_title,
            "keywords": keywords
        }
        line = "标题："+ news_title + "\n 类别：" + category_name
        line = [w for w in line]
        sents = chunks(line, max_len)
        if len(sents[-1]) < min_len:  # the last one is too short
            sents = sents[:-1]
        data.extend(sents)
    return data


class DataLoader_classification(object):
    def __init__(self, tknizer, filename, batch_size, max_len, min_len):
        self.batch_size = batch_size
        self.tknizer = tknizer
        self.max_len = max_len
        self.min_len = min_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

    def __iter__(self):
        lines = self.stream.readlines(BUFSIZE)

        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)

        data = parse_lines_classification(lines[:-1], self.max_len, self.min_len)
        random.shuffle(data)

        idx = 0
        while idx < len(data):
            yield batchify(data[idx:idx + self.batch_size], self.tknizer)
            idx += self.batch_size

class DataLoader(object):
    def __init__(self, tknizer, filename, batch_size, max_len, min_len):
        self.batch_size = batch_size
        self.tknizer = tknizer
        self.max_len = max_len
        self.min_len = min_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

    def __iter__(self):
        lines = self.stream.readlines(BUFSIZE)

        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)

        data = parse_lines(lines[:-1], self.max_len, self.min_len)  # the last sent may be imcomplete
        random.shuffle(data)

        idx = 0
        while idx < len(data):
            yield batchify(data[idx:idx + self.batch_size], self.tknizer)
            idx += self.batch_size