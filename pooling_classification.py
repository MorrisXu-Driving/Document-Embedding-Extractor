import logging
from copy import deepcopy
import swifter
import ipdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from nltk.tokenize import sent_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sklearn.model_selection import KFold
from transformers import AutoModel, AutoTokenizer
import sys

sys.path.append('..')
sys.path.append('/raid1/p3/jiahao/simsiam/runs/bert_nli_64seqlen/')

movie_review_path = '/home/jiahao/nlp/tokens'


class CNNPoolingClassifier(nn.Module):
    def __init__(self, d_model, n_class):
        super(CNNPoolingClassifier, self).__init__()
        c = d_model // 3
        self.cnn1 = nn.Conv2d(1, c, (2, d_model))
        self.cnn2 = nn.Conv2d(1, c, (3, d_model))
        self.cnn3 = nn.Conv2d(1, c, (4, d_model))
        
        self.pooling = lambda x: torch.max(x, dim=-1)[0]
        # self.pooling = lambda x: torch.max(x, dim=-1)[0] + torch.mean(x, dim = -1)

        self.cls = nn.Sequential(nn.Linear(d_model, d_model),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(d_model, d_model),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(d_model, n_class))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # size of x: [batch, num_seq, hidden]
        x = x.unsqueeze(1)  # [b,1,n,h]
        assert isinstance(x, torch.Tensor)
        output1 = self.pooling(self.cnn1(x).squeeze(-1))  # [b, c]
        output2 = self.pooling(self.cnn2(x).squeeze(-1))
        output3 = self.pooling(self.cnn3(x).squeeze(-1))

        doc_embedding = torch.relu(torch.cat([output1, output2, output3], dim=1))  # [b, d_model]
        output = self.cls(doc_embedding)
        return output


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


import os


def prepare_movie_review():
    def helper(s):
        s = s.strip()
        s = ' '.join(s.split())
        s = sent_tokenize(s)
        return s

    def read_moive_review(path):
        data, target = [], []

        for file in os.listdir(path + '/neg'):
            with open(path + '/neg/' + file, 'r', encoding = 'ISO-8859-1') as open_file:
                txt = open_file.read()
                data.append(txt)
                target.append(0)

        for file in os.listdir(path + '/pos'):
            with open(path + '/pos/' + file, 'r', encoding = 'ISO-8859-1') as open_file:
                txt = open_file.read()
                data.append(txt)
                target.append(1)
        df = pd.DataFrame({'data': data, 'target': target})
        df = df.sample(frac=1).reset_index(drop=True)
        return df

    df = read_moive_review(movie_review_path)
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


def train_cnn(cls, embedding_bank, target_bank, n_epoch, batch_size=1, lr=1e-3, patience=5):
    assert isinstance(cls, nn.Module)
    cls = cls.train()
    cls = cls.to(device)

    optimizer = optim.Adam(cls.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    min_loss = 1e6
    patience_count = 0
    for epoch in tqdm(range(n_epoch), desc='train'):
        step = 0
        epoch_loss = 0
        for i in range(0, len(embedding_bank), batch_size):
            step += 1
            embeddings = embedding_bank[i:i + batch_size]
            targets = target_bank[i:i + batch_size]
            input_tensor = torch.tensor(embeddings, dtype=torch.float).to(device)
            output_tensor = cls(input_tensor)  # (b, n_class)
            target_tensor = torch.tensor(targets).to(device)
            loss = criterion(output_tensor, target_tensor)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_model = deepcopy(cls).cpu()
        else:
            patience_count += 1
            if patience_count == patience:
                print('early stopping')
                break
    return best_model


def eval_cnn(cls, embedding_bank, target_bank, batch_size=1, max_num_seq=64):
    assert isinstance(cls, nn.Module)
    cls = cls.to(device)
    cls = cls.eval()
    for p in cls.parameters():
        p.requires_grad = False
    predicted_bank = []
    with torch.no_grad():
        for i in tqdm(range(0, len(embedding_bank), batch_size), desc='eval'):
            embeddings = embedding_bank[i:i + batch_size]
            input_tensor = torch.tensor(embeddings, dtype=torch.float).to(device)
            output_tensor = cls(input_tensor)  # (b, n_class)
            assert isinstance(output_tensor, torch.Tensor)
            predicted = output_tensor.argmax(dim=-1)  # (b)
            predicted_bank += predicted.tolist()

    p, r, f, _ = precision_recall_fscore_support(target_bank, predicted_bank, average='macro')
    pp, rr, ff, _ = precision_recall_fscore_support(target_bank, predicted_bank, average='micro')
    return p, r, f, pp, rr, ff


def document_embedding(model, doc, tokenizer, embedding_batch_size=64):
    embedding = []
    output_features = tokenizer(doc, padding='max_length', truncation=True, return_tensors='pt')
    n_sents = len(output_features.data['input_ids'])
    for i in range(0, n_sents, embedding_batch_size):
        input_ids = output_features.data['input_ids'][i:i + embedding_batch_size]
        length = output_features.data['attention_mask'][i:i + embedding_batch_size].sum(dim=-1)
        if is_simcse:
            sent_embeddings = model(input_ids.to(device), return_dict=True).pooler_output.cpu()
        else:
            sent_embeddings = model(input_ids.to(device), length).cpu()
        embedding.append(sent_embeddings)
    embedding = torch.cat(embedding, dim=0)
    return embedding.numpy()


def get_sent_embedding(docs):
    embedding_bank = []
    if is_pretrained:
        # pretrained sbert
        if model_type in ['nli-bert-base', 'stsb-bert-base']:
            # pretrained sentence bert
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_type)
            model = model.eval()
            model = model.to(device)
            for p in model.parameters():
                p.requires_grad = False
            with torch.no_grad():
                for doc in tqdm(docs, desc='embedding'):
                    embedding_bank.append(model.encode(doc, show_progress_bar=False))
        elif model_type in ['bert-base-uncased']:
            # pertrained bert
            model = AutoModel.from_pretrained(model_type)
            tokenizer = AutoTokenizer.from_pretrained(model_type)
            model = model.eval()
            model = model.to(device)
            for p in model.parameters():
                p.requires_grad = False
            with torch.no_grad():
                for doc in tqdm(docs, desc='embedding'):
                    embedding_bank.append(document_embedding(model, doc, tokenizer))
    else:
        if is_simcse:
            # self-trained simcse
            model = AutoModel.from_pretrained(model_path)
        else:
            # self-trained BlendCL-RST
            model = load_from(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = model.eval()
        model = model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        with torch.no_grad():
            for doc in tqdm(docs, desc='embedding'):
                embedding_bank.append(document_embedding(model, doc, tokenizer))
    return embedding_bank

def convert_load_model(model, input=torch.tensor(torch.rand(size=(1,3,112,112)))):
    model = torch.jit.trace(model, input)
    torch.jit.save(model, cp_file_path + '/model.tjm')
    model = torch.jit.load(cp_file_path + '/model.tjm')
    return model

def get_embedding_bank(dataset, max_num_seq):
    if dataset == '20ng':
        input, target = prepare_20ng()
    elif dataset == 'reuters':
        input, target = prepare_reuters()
    elif dataset == 'mr':
        input, target = prepare_movie_review()
    else:
        raise NotImplementedError

    n_class = target.max() + 1
    if is_debug:
        input, target = input[:100], target[:100]
    embedding = get_sent_embedding(input)
    embedding = formatting(embedding, max_num_seq)  # (N, n_sentences, word_embedding_dim)
    return embedding, np.array(target), n_class


def dataset_training(dataset):
    logging.info('Training on {}'.format(dataset))
    P, R, F, PP, RR, FF = 0, 0, 0, 0, 0, 0
    if dataset == '20ng':
        max_num_seq = 32
    elif dataset == 'reuters':
        max_num_seq = 32
    elif dataset == 'mr':
        max_num_seq = 64
    else:
        raise NotImplementedError
    embedding, target, n_class = get_embedding_bank(dataset, max_num_seq)
    logging.info('{} has {} class'.format(dataset, n_class))
    kf = KFold(n_fold)
    for fold, (train_index, dev_index) in enumerate(kf.split(embedding)):
        train_embedding, train_target = embedding[train_index], target[train_index]
        dev_embedding, dev_target = embedding[dev_index], target[dev_index]
        cnn_classifier = CNNPoolingClassifier(768, n_class)
        cnn_classifier = train_cnn(cnn_classifier, train_embedding, train_target, n_epoch, batch_size, lr, patience)
        p, r, f, pp, rr, ff = eval_cnn(cnn_classifier, dev_embedding, dev_target, batch_size, max_num_seq)
        P += p
        R += r
        F += f
        PP += pp
        RR += rr
        FF += ff
        logging.info(
            'Fold{}::MACRO p:{:.4f}-r:{:.4f}-f1:{:.4f}::MICRO p:{:.4f}-r:{:.4f}-f1:{:.4f}'.format(fold, p, r, f, pp, rr,
                                                                                                  ff))
        print(
            'Fold{}::MACRO p:{:.4f}-r:{:.4f}-f1:{:.4f}::MICRO p:{:.4f}-r:{:.4f}-f1:{:.4f}'.format(fold, p, r, f, pp, rr,
                                                                                                  ff))
    P /= n_fold
    R /= n_fold
    F /= n_fold
    PP /= n_fold
    RR /= n_fold
    FF /= n_fold
    logging.info(
        'Ave::MACRO p:{:.4f}-r:{:.4f}-f1:{:.4f}::MICRO p:{:.4f}-r:{:.4f}-f1:{:.4f}'.format(P, R, F, PP, RR, FF))
    print('Ave::MACRO p:{:.4f}-r:{:.4f}-f1:{:.4f}::MICRO p:{:.4f}-r:{:.4f}-f1:{:.4f}'.format(P, R, F, PP, RR, FF))


def main():
    for task in tasks:
        dataset_training(task)


if __name__ == '__main__':

    is_pretrained = False
    is_simcse = False
    # model_path = "/raid1/p3/jiahao/simsiam/runs/my-sup-simcse-bert-base-uncased"

    # pclm_type = 'albert-base-v1'
    # pclm_type = 'bert-base-uncased'
    # pclm_type = 'nli-bert-base'
    # pclm_type = 'stsb-bert-base'

    # model_type='albert-base-v2'
    model_type = 'bert-base-uncased'


    # model_path = '/raid1/p3/jiahao/simsiam/runs/pclm/albert/checkpoints.cp'
    # model_path = '/raid1/p3/jiahao/simsiam/runs/pclm/bert/checkpoints.cp'
    # model_path = '/raid1/p3/jiahao/simsiam/runs/sbert/nli_bert_base/checkpoints.cp'
    # model_path = '/raid1/p3/jiahao/simsiam/runs/sbert/stsb_bert_base/checkpoints.cp'
    # model_path = '/raid1/p3/jiahao/simsiam/runs/albert_nli_64seqlen/checkpoints.cp'
    # model_path = '/raid1/p3/jiahao/simsiam/runs/albert_nli_64seqlen_on_stsb/checkpoints.cp'
    model_path = '/raid1/p3/jiahao/simsiam/runs/bert_nli_64seqlen/checkpoints.cp'
    # model_path = '/raid1/p3/jiahao/simsiam/runs/bert_nli_64seqlen_on_stsb/checkpoints.cp'
    
    if is_pretrained:
        output_path = './logs/cnn/classification/{}'.format(model_type)
    elif is_simcse:
        output_path = './logs/cnn/classification/{}'.format(model_path.split('/')[-1])
    else:
        output_path = './logs/cnn/classification/{}_{}'.format(model_path.split('/')[-3], model_path.split('/')[-2])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(filename=output_path + '/max_pooling_test.log', level=logging.INFO)
    import time
    logging.info('''-------------------------------------------------
    {}
    ------------------------------------------------'''.format(time.asctime(time.localtime(time.time()))))
    device = torch.device('cuda:2')
    is_debug = False
    n_fold = 5
    n_epoch = 1000
    batch_size = 32
    lr = 1e-3

    patience = 5
    tasks = ['20ng', 'reuters', 'mr']
    # tasks = ['reuters']
    main()
