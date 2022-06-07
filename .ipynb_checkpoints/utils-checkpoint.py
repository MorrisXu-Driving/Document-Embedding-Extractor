import logging
import numpy as np
import pandas as pd
import swifter
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import sent_tokenize

def load_from(cp_file_path):
    print('loading from {}'.format(cp_file_path))
    cp_file = torch.load(cp_file_path)
    model = cp_file['simsiam'].backbone
    return model

def prepare_20ng():
    logging.info('Eval on 20 News Group')

    def helper(s):
        s = s.strip()
        s = ' '.join(s.split())
        s = sent_tokenize(s)
        return s

    dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data, target = dataset['data'], dataset['target']
    df = pd.DataFrame({'data': data, 'target': target})
    df['data'].replace('', np.nan, inplace=True)
    df.dropna(subset=['data'], inplace=True)
    df['texts'] = df['data'].swifter.apply(helper)
    df = df[df.astype(str)['texts'] != '[]']
    df['num_sents'] = df['texts'].swifter.apply(len)
    ave_n_sents = df['num_sents'].to_numpy().mean()
    logging.info('Ave number of sentences:{:.4f}'.format(ave_n_sents))
    return df['texts'].tolist(), df['target'].to_numpy()


def prepare_reuters():
    def helper(s):
        s = s.strip()
        s = ' '.join(s.split())
        s = sent_tokenize(s)
        return s

    from nltk.corpus import reuters
    from sklearn.preprocessing import LabelEncoder
    text = []
    cats = []
    for index, i in enumerate(reuters.fileids()):
        cat = reuters.categories(fileids=[i])
        if len(cat) == 1:
            text.append(reuters.raw(fileids=[i]))
            cats.extend(cat)
    le = LabelEncoder()
    le.fit(cats)
    label = le.transform(cats)
    df = pd.DataFrame({'data': text, 'target': label})
    df['data'].replace('', np.nan, inplace=True)
    df.dropna(subset=['data'], inplace=True)
    df['texts'] = df['data'].swifter.apply(helper)
    df = df[df.astype(str)['texts'] != '[]']
    df['num_sents'] = df['texts'].swifter.apply(len)
    ave_n_sents = df['num_sents'].to_numpy().mean()
    logging.info('Ave number of sentences:{:.4f}'.format(ave_n_sents))
    return df['texts'].tolist(), df['target'].to_numpy()


def formatting(x, max_num_seq):
    print('formatting...')
    embedding_size = len(x[0][0])
    res = []
    for xx in x:
        if len(xx) >= max_num_seq:
            res.append(xx[:max_num_seq])
        else:
            tmp = np.zeros((max_num_seq - len(xx), embedding_size))
            res.append(np.concatenate((xx, tmp), axis=0))
    print('done')
    return np.stack(res, axis=0)
    # x: list of list