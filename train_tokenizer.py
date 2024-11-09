import json
from multiprocessing import Pool
from collections import Counter
from tqdm import tqdm
BUFSIZE=10000

ttype = 'char'

# 将传入的文档 doc 拆分成单个字符的列表
def process(doc):
    res = [w for w in doc]
    return res

# 使用多线程的方式并行处理文档列表 docs，将每个文档通过 process 函数处理后，将结果汇总并更新到一个计数器 cnt 中
def save(cnt, docs, nprocessors):
    res = pool.map(process,docs, len(docs)//nprocessors)
    all_lines =[]
    for xs in res:
        all_lines.extend(xs)
    for x in all_lines:
        cnt.update(x)


if ttype =='char':
    # 初始化一个Counter对象cnt用于计数
    cnt = Counter()
    nprocessors=1000
    pool = Pool(nprocessors)
    docs = []
    with open('/mnt/share/xujing/nuaa_nlp/data/data/train.txt', 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            line=json.loads(line)['text']
            if not line:
                continue
            docs.append(line)

            if len(docs) == BUFSIZE:
                save(cnt, docs, nprocessors)
                docs = []
                print(BUFSIZE)
        if len(docs) > 0:
            save(cnt, docs, nprocessors)
            print(len(docs))

    print("vocab")
    with open("/mnt/share/xujing/nuaa_nlp/model/vocab.txt", 'w',encoding ='utf8') as f:
        for x, y in cnt.most_common():
            f.write(x + '\t' + str(y) + '\n')
        print("done")

elif ttype == 'bpe':

    import sentencepiece as spm

    spm.SentencePieceTrainer.train(input='/mnt/share/xujing/nuaa_nlp/data/data/train.txt',model_prefix='m',vocab_size=24000,
                                   character_coverage = 1.0, model_type = 'bpe',
                                   user_defined_symbols=['<pad>','<bos>' ,'<eos>','<mask>','<INST>','</INST>', '<SYS>', '</SYS>'])
else:
    assert "Unsupport type."
    pass