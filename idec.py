import argparse
import logging
from copy import deepcopy
import swifter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import transformers
from RetMetric import RetMetric
from nltk.tokenize import sent_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sklearn.model_selection import KFold
from transformers import AutoModel, AutoTokenizer
import ipdb
import sys
import os
sys.path.append('..')

movie_review_path = '/raid1/p3/jiahao/Datasets/cornell_movie_review'
fake_news_path = "/raid1/p3/jiahao/Datasets/fake_news_detection"
corona_tweets_path="/raid1/p3/jiahao/Datasets/corona_tweets/Corona_NLP"

class IDEC(nn.Module):
    def __init__(self, d_model, n_emb, n_cluster, alpha = 1):
        super(IDEC, self).__init__()
        self.alpha = alpha
        self.ae = CNNPoolingAE(d_model, n_emb)
        self.cluster_layer = nn.Parameter(torch.Tensor(n_cluster, n_emb))
        torch.nn.init.xavier_uniform_(self.cluster_layer.data)
    
    def forward(self, x, n_sents=None):
        x_prime, z = self.ae(x)
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_prime, q

        
class CNNPoolingAE(nn.Module):
    def __init__(self, d_model, n_emb):
        super(CNNPoolingAE, self).__init__()
        c = d_model // 3
        
        self.encoder = CNNPoolingClassifier(d_model, n_emb)
        
        self.tcnn1 = nn.Sequential(nn.ConvTranspose2d(c, 1, (2, d_model), padding=[1,0]), nn.ReLU(inplace=True))
        self.tcnn2 = nn.Sequential(nn.ConvTranspose2d(c, 1, (3, d_model), padding=[1,0]), nn.ReLU(inplace=True))
        self.tcnn3 = nn.Sequential(nn.ConvTranspose2d(c, 1, (4, d_model), padding=[1,0]), nn.ReLU(inplace=True))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def pretrain(self, path = ''):
        if path == '':
            pretrain_ae(self.ae)

    def forward(self, x, n_sents=None):
        z, f1, f2, f3 = self.encoder(x)
        f1 = self.tcnn1(f1)
        f2 = self.tcnn2(f2)
        f3 = self.tcnn3(f3)
        x_prime = f1 + f2 + f3
        return x_prime, z


class CNNPoolingClassifier(nn.Module):
    def __init__(self, d_model, n_emb):
        super(CNNPoolingClassifier, self).__init__()
        c = d_model // 3
        self.cnn1 = nn.Conv2d(1, c, (2, d_model), padding=[1,0])
        self.cnn2 = nn.Conv2d(1, c, (3, d_model), padding=[1,0])
        self.cnn3 = nn.Conv2d(1, c, (4, d_model), padding=[1,0])

        self.pooling = lambda x: torch.max(x, dim=-1)[0]

        self.cls = nn.Sequential(nn.Linear(d_model, d_model),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(d_model, d_model),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(d_model, n_emb))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, n_sents=None):
        # size of x: [batch, num_seq, hidden]
        x = x.unsqueeze(1)  # [b,1,n,h]
        assert isinstance(x, torch.Tensor)

        f1 = self.cnn1(x)  # [b, c, n, 1]
        f2 = self.cnn2(x)  # [b, c, n, 1]
        f3 = self.cnn3(x)  # [b, c, n, 1]

        doc_embedding = torch.relu(torch.cat([self.pooling(f1.squeeze(-1)), self.pooling(f2.squeeze(-1)), self.pooling(f3.squeeze(-1))], dim=1))  # [b, d_model]
        output = self.cls(doc_embedding)
        return output, f1, f2, f3

    
import datasets
from datasets import load_dataset
def prepare_fake_news_detection(args):
    logging.info("eval on Sentiment140")

    dataset = load_dataset("csv", data_files=fake_news_path+"/True.csv")["train"]
    pos_dataset = dataset.map(lambda example: {"label":[1]*len(example[dataset.column_names[0]])}, batched=True)
    dataset = load_dataset("csv", data_files=fake_news_path+"/Fake.csv")["train"]
    neg_dataset = dataset.map(lambda example:{"label":[0]*len(example[dataset.column_names[0]])}, batched=True)
    combined=datasets.concatenate_datasets([pos_dataset, neg_dataset]).shuffle(seed=42)
    def helper(s):
        s = s.strip()
        s = ' '.join(s.split())
        s = sent_tokenize(s)
        return s

    combined = combined.map(lambda example: {"sents":[helper(s) for s in example["text"]]}, batched=True)
    df = pd.DataFrame({'data': combined["sents"], 'target':combined["label"] })
    if args.is_debug:
        df =  df[:100]

    # filter out the empty list
    df = df[df["data"].apply(lambda x: len(x)>0)]


    return df["data"].tolist(), df["target"].to_numpy()



def helper(s):
    s = s.strip()
    s = ' '.join(s.split())
    s = sent_tokenize(s)
    return s

def prepare_ag_news(args):

    logging.info("eval on AG news")
    dataset = load_dataset("ag_news")["train"]
    dataset = dataset.map(lambda examples:{"data":[helper(s) for s in examples["text"]]})
    df = pd.DataFrame({"data": dataset["data"],
                       "label": dataset["label"]})

    if args.is_debug:
        df = df[:100]
    df = df[df["data"].apply(lambda x: len(x)>0)]
    return df["data"].tolist(), df["target"].to_numpy()

def prepare_corona_tweets(args):
    logging.info("eval on corona twittes")

    dataset = load_dataset("csv", data_files=corona_tweets_path+"_train.csv", encoding="ISO-8859-1")["train"]
    dataset = datasets.concatenate_datasets([dataset, load_dataset("csv", data_files=corona_tweets_path+"_test.csv", encoding="ISO-8859-1")["train"]])
    # dataset = dataset.map(lambda examples:{"cleaned":[cleaning(s) for s in examples["OriginalTweet"]]}, batched=True)
    dataset = dataset.map(lambda examples: {"sents":[helper(s) for s in examples["OriginalTweet"]]}, batched=True)

    def labeling(examples):
        res=[]
        for label in examples["Sentiment"]:
            if "Positive" in label:
                res.append(2)
            elif "Negative" in label:
                res.append(0)
            elif "Neutral" in label:
                res.append(1)
            else:
                raise NotImplementedError
        return {"target":res}
    dataset = dataset.map(labeling, batched=True)
    df = pd.DataFrame({"data": dataset["sents"], "target":dataset["target"]})

    if args.is_debug:
        df = df[:100]

    return df["data"].tolist(), df["target"].to_numpy()


def prepare_20ng(args):
    logging.info('Eval on 20 News Group')

    def helper(s):
        s = s.strip()
        s = ' '.join(s.split())
        s = sent_tokenize(s)
        return s

    dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data, target = dataset['data'], dataset['target']
    df = pd.DataFrame({'data': data, 'target': target})
    if args.is_debug:
        df = df[:100]

    df['data'].replace('', np.nan, inplace=True)
    df.dropna(subset=['data'], inplace=True)
    df['texts'] = df['data'].swifter.apply(helper)
    df = df[df.astype(str)['texts'] != '[]']
    df['num_sents'] = df['texts'].swifter.apply(len)
    ave_n_sents = df['num_sents'].to_numpy().mean()
    logging.info('Ave number of sentences:{:.4f}'.format(ave_n_sents))
    return df['texts'].tolist(), df['target'].to_numpy()


import os


def prepare_movie_review(args):
    def helper(s):
        s = s.strip()
        s = ' '.join(s.split())
        s = sent_tokenize(s)
        return s

    def read_moive_review(path):
        data, target = [], []

        for file in os.listdir(path + '/neg'):
            with open(path + '/neg/' + file, 'r') as open_file:
                txt = open_file.read()
                data.append(txt)
                target.append(0)

        for file in os.listdir(path + '/pos'):
            with open(path + '/pos/' + file, 'r') as open_file:
                txt = open_file.read()
                data.append(txt)
                target.append(1)
        df = pd.DataFrame({'data': data, 'target': target})
        df = df.sample(frac=1).reset_index(drop=True)
        return df

    df = read_moive_review(movie_review_path)
    if args.is_debug:
        df = df[:100]
    df['data'].replace('', np.nan, inplace=True)
    df.dropna(subset=['data'], inplace=True)
    df['texts'] = df['data'].swifter.apply(helper)
    df = df[df.astype(str)['texts'] != '[]']
    df['num_sents'] = df['texts'].swifter.apply(len)
    ave_n_sents = df['num_sents'].to_numpy().mean()
    logging.info('Ave number of sentences:{:.4f}'.format(ave_n_sents))
    return df['texts'].tolist(), df['target'].to_numpy()


def prepare_reuters(args):
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
    if args.is_debug:
        df = df[:100]
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

def get_sent_embedding(docs, args):
    embedding_bank = []

    if args.model_type in ["plm", "simcse"]:
        model = AutoModel.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.model_type=="sbert":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model_name_or_path)
    elif args.model_type=="blendrst":
        from models import Backbone
        if "albert" in args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained("albert-base-uncased")
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = torch.load(args.model_name_or_path)["simsiam"].backbone

    # preparing the model
    model = model.eval().cuda()
    for p in model.parameters():
        p.requires_grad=False

    # preparing the documents
    def flatten_doc2sent(docs):
        """flattent docs from a list of list into sentence list"""
        sents,n_sents=[],[]
        for doc in docs:
            n_sents.append(len(doc))
            sents.extend(doc)
        return sents, n_sents

    def unflatten_sent2doc(sents, n_sents):
        """unflatten the sentence representation into document Nxd matrix"""
        doc_embeddings=[]
        while len(n_sents)>0:
            n_sent = n_sents.pop(0)
            doc_embeddings.append(np.stack(sents[:n_sent], axis=0))
            del sents[:n_sent]
        return doc_embeddings

    sents, n_sents = flatten_doc2sent(docs)


    # embedding
    def embedding():
        embedding_bank=[]
        n_step = len(sents)//args.batch_size
        if len(sents)%args.batch_size!=0:
            n_step+=1
        for step in tqdm(range(n_step)):
            samples = sents[step*args.batch_size:(step+1)*args.batch_size]
            if args.model_type=="sbert":
                sent_embeddings=model.encode(samples, show_progress_bar=False)
            else:
                batch = tokenizer(samples, max_length=args.max_seq_length, padding=True, truncation=True, return_tensors="pt").to("cuda")
                if args.model_type in ["simcse", "plm"]:
                    if args.model_type=="simcse":
                        sent_embeddings = model(**batch, return_dict=True).pooler_output
                    else:
                        sent_embeddings = ((model(**batch).last_hidden_state)*(batch["attention_mask"].unsqueeze(-1))).sum(dim=1)/batch["attention_mask"].sum(dim=-1, keepdim=True)
                else:
                    sent_embeddings = model(batch["input_ids"], batch["attention_mask"].sum(dim=-1))
                sent_embeddings = sent_embeddings.cpu().numpy()
            embedding_bank.extend(sent_embeddings)
        return embedding_bank


    with torch.no_grad():
        embedding_bank = embedding()

    embedding_bank = unflatten_sent2doc(embedding_bank, n_sents)
    return embedding_bank


def get_embedding_bank(dataset, args):
    if dataset == '20ng':
        input, target = prepare_20ng(args)
    elif dataset == 'reuters':
        input, target = prepare_reuters(args)
    elif dataset == 'mr':
        input, target = prepare_movie_review(args)
    elif dataset =="fake_news":
        input, target = prepare_fake_news_detection(args)
    elif dataset =="corona":
        input, target = prepare_corona_tweets(args)
    else:
        raise NotImplementedError

    n_class = target.max() + 1
    if args.is_debug:
        input, target = input[:100], target[:100]
    embedding = get_sent_embedding(input, args)
    return embedding, np.array(target), n_class

def aggregator(x, n_sents, strategy="minmax"):

    bs, seq_len, d = x.size()
    attention_mask = torch.zeros([bs, seq_len], dtype=torch.int, device=x.device)
    for i,n in enumerate(n_sents):
        attention_mask[i,:n]=1

    if strategy=="mean":
        return (x*attention_mask.unsqueeze(-1)).sum(dim=1)/attention_mask.sum(dim=-1, keepdim=True)
    elif strategy=="max":
        return torch.max((x-10000*torch.logical_not(attention_mask).unsqueeze(-1)), dim=1)[0]
    elif strategy=="min":
        return torch.min((x+10000*torch.logical_not(attention_mask).unsqueeze(-1)), dim=1)[0]
    elif strategy =="minmax":
        return torch.cat([torch.max((x-10000*torch.logical_not(attention_mask).unsqueeze(-1)), dim=1)[0],
               torch.min((x+10000*torch.logical_not(attention_mask).unsqueeze(-1)), dim=1)[0]], dim=-1)
    else:
        raise NotImplementedError


from sklearn import cluster
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
class Confusion(object):
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """

    def __init__(self, k, normalized=False):
        super(Confusion, self).__init__()
        self.k = k
        self.conf = torch.LongTensor(k, k)
        self.normalized = normalized
        self.conf.fill_(0)
        self.gt_n_cluster = None

    def cuda(self):
        self.conf = self.conf.cuda()

    def add(self, output, target):
        output = output.squeeze()
        target = target.squeeze()
        assert output.size(0) == target.size(0), \
            'number of targets and outputs do not match'
        if output.ndimension() > 1:  # it is the raw probabilities over classes
            assert output.size(1) == self.conf.size(0), \
                'number of outputs does not match size of confusion matrix'

            _, pred = output.max(1)  # find the predicted class
        else:  # it is already the predicted class
            pred = output
        indices = (target * self.conf.stride(0) + pred.squeeze_().type_as(target)).type_as(self.conf)
        ones = torch.ones(1).type_as(self.conf).expand(indices.size(0))
        self._conf_flat = self.conf.view(-1)
        self._conf_flat.index_add_(0, indices, ones)

    def classIoU(self, ignore_last=False):
        confusion_tensor = self.conf
        if ignore_last:
            confusion_tensor = self.conf.narrow(0, 0, self.k - 1).narrow(1, 0, self.k - 1)
        union = confusion_tensor.sum(0).view(-1) + confusion_tensor.sum(1).view(-1) - confusion_tensor.diag().view(-1)
        acc = confusion_tensor.diag().float().view(-1).div(union.float() + 1)
        return acc

    def recall(self, clsId):
        i = clsId
        TP = self.conf[i, i].sum().item()
        TPuFN = self.conf[i, :].sum().item()
        if TPuFN == 0:
            return 0
        return float(TP) / TPuFN

    def precision(self, clsId):
        i = clsId
        TP = self.conf[i, i].sum().item()
        TPuFP = self.conf[:, i].sum().item()
        if TPuFP == 0:
            return 0
        return float(TP) / TPuFP

    def f1score(self, clsId):
        r = self.recall(clsId)
        p = self.precision(clsId)
        print("classID:{}, precision:{:.4f}, recall:{:.4f}".format(clsId, p, r))
        if (p + r) == 0:
            return 0
        return 2 * float(p * r) / (p + r)

    def acc(self):
        TP = self.conf.diag().sum().item()
        total = self.conf.sum().item()
        if total == 0:
            return 0
        return float(TP) / total

    def optimal_assignment(self, gt_n_cluster=None, assign=None):
        if assign is None:
            mat = -self.conf.cpu().numpy()  # hungaian finds the minimum cost
            r, assign = hungarian(mat)
        self.conf = self.conf[:, assign]
        self.gt_n_cluster = gt_n_cluster
        return assign

    def show(self, width=6, row_labels=None, column_labels=None):
        print("Confusion Matrix:")
        conf = self.conf
        rows = self.gt_n_cluster or conf.size(0)
        cols = conf.size(1)
        if column_labels is not None:
            print(("%" + str(width) + "s") % '', end='')
            for c in column_labels:
                print(("%" + str(width) + "s") % c, end='')
            print('')
        for i in range(0, rows):
            if row_labels is not None:
                print(("%" + str(width) + "s|") % row_labels[i], end='')
            for j in range(0, cols):
                print(("%" + str(width) + ".d") % conf[i, j], end='')
            print('')

    def conf2label(self):
        conf = self.conf
        gt_classes_count = conf.sum(1).squeeze()
        n_sample = gt_classes_count.sum().item()
        gt_label = torch.zeros(n_sample)
        pred_label = torch.zeros(n_sample)
        cur_idx = 0
        for c in range(conf.size(0)):
            if gt_classes_count[c] > 0:
                gt_label[cur_idx:cur_idx + gt_classes_count[c]].fill_(c)
            for p in range(conf.size(1)):
                if conf[c][p] > 0:
                    pred_label[cur_idx:cur_idx + conf[c][p]].fill_(p)
                cur_idx = cur_idx + conf[c][p];
        return gt_label, pred_label

    def clusterscores(self):
        target, pred = self.conf2label()
        NMI = normalized_mutual_info_score(target, pred)
        ARI = adjusted_rand_score(target, pred)
        AMI = adjusted_mutual_info_score(target, pred)
        return {'NMI': NMI, 'ARI': ARI, 'AMI': AMI}


'''non-parametric k-means clustering'''
def clustering_single_trial(y_true, embeddings=None, num_classes=10, random_state=0):
    """"Evaluate the embeddings using KMeans"""
    kmeans = cluster.KMeans(n_clusters=num_classes, random_state=random_state)
    kmeans.fit(embeddings.cpu().detach().numpy())
    y_pred = kmeans.labels_.astype(np.int)

    confusion = Confusion(num_classes)
    confusion.add(torch.tensor(y_pred), torch.tensor(y_true))
    confusion.optimal_assignment(num_classes)
    return y_pred, kmeans, confusion.acc(), confusion.clusterscores()



def train_cluster(n_class, dev_embedding, dev_n_sents, dev_target):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding = aggregator(x=torch.tensor(dev_embedding, dtype=torch.float, device=device),
                           n_sents=torch.tensor(dev_n_sents, dtype=torch.int, device=device)).cpu().numpy()
    _, _, acc, scores=clustering_single_trial(dev_target, embedding, num_classes=n_class)
    scores["ACC"]=acc
    return None, scores

'''parametric CNN pooler for classification'''
def train_kfold_classifier(args, n_class, embedding, n_sents, target):
    kf = KFold(args.n_fold)
    metrics = []
    for fold, (train_index, dev_index) in enumerate(kf.split(embedding)):
        train_embedding, train_n_sents, train_target = embedding[train_index], n_sents[train_index], target[train_index]
        dev_embedding, dev_n_sents, dev_target = embedding[dev_index], n_sents[dev_index], target[dev_index]
        cnn_classifier, eval_metric = train_classifier(n_class, train_embedding, train_n_sents, train_target,
                                                       dev_embedding, dev_n_sents, dev_target,
                                                       args.n_epoch, args.batch_size, args.lr, args.patience)
        metrics.append(eval_metric)

    avg_metric = {k: 0 for k in metrics[0].keys()}
    for key in avg_metric.keys():
        avg_metric[key] = sum([metrics[i][key] for i in range(len(metrics))]) / args.n_fold
    return avg_metric


def train_classifier(n_class, train_embedding, train_n_sents, train_target,
                     dev_embedding, dev_n_sents, dev_target,
                     n_epoch, batch_size=1, lr=1e-3, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def eval(cls, dev_embedding, dev_n_sents, dev_target, batch_size=1):
        assert isinstance(cls, nn.Module)
        cls = cls.to(device)
        cls = cls.eval()
        for p in cls.parameters():
            p.requires_grad = False
        predicted_bank = []
        with torch.no_grad():
            for i in tqdm(range(0, len(dev_embedding), batch_size), desc='eval'):
                embeddings = torch.tensor(dev_embedding[i:i + batch_size], dtype=torch.float, device=device)
                n_sents = torch.tensor(dev_n_sents[i:i + batch_size], dtype=torch.int, device=device)

                output_tensor, _, _, _ = cls(embeddings, n_sents)  # (b, n_class)
                assert isinstance(output_tensor, torch.Tensor)
                predicted = output_tensor.argmax(dim=-1)  # (b)
                predicted_bank += predicted.tolist()

        p, r, f, _ = precision_recall_fscore_support(dev_target, predicted_bank, average='macro')
        pp, rr, ff, _ = precision_recall_fscore_support(dev_target, predicted_bank, average='micro')
        res = {"p": p, "r": r, "f": f,
               "pp": pp, "rr": rr, "ff": ff}
        return res

    cls = CNNPoolingClassifier(768, n_class)
    assert isinstance(cls, nn.Module)
    cls = cls.train()
    cls = cls.to(device)

    optimizer = optim.Adam(cls.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    min_loss = 1e6
    patience_count = 0

    # eval_metric = eval(cls, dev_embedding, dev_n_sents, dev_target, batch_size)
    for epoch in tqdm(range(n_epoch), desc='train'):
        step = 0
        epoch_loss = 0
        for i in range(0, len(train_embedding), batch_size):
            step += 1
            embeddings = torch.tensor(train_embedding[i:i + batch_size], dtype=torch.float, device=device)
            n_sents = torch.tensor(train_n_sents[i:i+batch_size], dtype=torch.int, device=device)
            targets = torch.tensor(train_target[i:i + batch_size], device=device)

            output_tensor, _, _, _ = cls(embeddings, n_sents)  # (b, n_class)
            loss = criterion(output_tensor, targets)

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

    eval_metric = eval(cls, dev_embedding, dev_n_sents, dev_target, batch_size)
    return best_model, eval_metric


'''IDEC'''
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def pretrain(args, ae, embedding, pretrain_path='', patience = 5):
    if path == '':
        pretrain_path = args.output_path
        ae = pretrain_ae(args, ae, embedding, pretrain_path, patience)
    else:
        ae.load_state_dict(torch.load(pretrain_path))
        print(f'load pretrained ae from {pretrain_path}')
    return ae
        
def pretrain_ae(args, ae, train_embedding, pretrain_path, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(ae.parameters(), lr = args.lr)
    batch_size = args.batch_size
    min_loss = 1e6
    patience_count = 0
    for epoch in tqdm(range(args.n_epoch), desc='pretrain_ae_train'):
        epoch_loss = 0.
        for i in range(0, len(train_embedding), args.batch_size):
            x = torch.tensor(train_embedding[i:i + batch_size], dtype=torch.float, device=device)
            optimizer.zero_grad()
            x_prime, _ = ae(x)
            loss = F.mse_loss(x_prime.squeeze(1), x)  # [128, 1, 32, 768]
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        if args.n_epoch % (args.n_epoch / 5) == 0:
            print("epoch {} loss = {:.4f}".format(epoch, epoch_loss / i))
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_model = deepcopy(ae.state_dict())
        else:
            patience_count += 1
            if patience_count == patience:
                print('early stop')
                break

    torch.save(ae.state_dict(), pretrain_path + '/ae.pth')
    print(f"pretrained and saved model to {pretrain_path}.")
    return ae


def train_kfold_idec(args, n_class, embedding, n_sents, target):
    kf = KFold(args.n_fold)
    metrics = []
    for fold, (train_index, dev_index) in enumerate(kf.split(embedding)):
        train_embedding, train_n_sents, train_target = embedding[train_index], n_sents[train_index], target[train_index]
        dev_embedding, dev_n_sents, dev_target = embedding[dev_index], n_sents[dev_index], target[dev_index]
        cnn_idec, eval_metric = train_idec(args, n_class, train_embedding, train_n_sents, train_target,
                                                       dev_embedding, dev_n_sents, dev_target,
                                                       args.n_epoch, args.batch_size, args.lr, args.patience)
        metrics.append(eval_metric)

    avg_metric = {k: 0 for k in metrics[0].keys()}
    for key in avg_metric.keys():
        avg_metric[key] = sum([metrics[i][key] for i in range(len(metrics))]) / args.n_fold
    return avg_metric
    
    
def train_idec(args, n_class, train_embedding, train_n_sents, train_target,
                     dev_embedding, dev_n_sents, dev_target,
                     n_epoch, batch_size=1, lr=1e-3, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def eval(model, dev_embedding, dev_n_sents, dev_target, y_pred_last, batch_size=1):
        from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
        from sklearn.metrics.cluster import adjusted_rand_score as ari
        from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
        from RetMetric import cluster_acc
        assert isinstance(model, nn.Module)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        step, ACC, NMI, ARI, AMI = 0, 0, 0, 0, 0
        with torch.no_grad():
            for i in tqdm(range(0, len(dev_embedding), batch_size), desc='eval'):
                step += 1
                embeddings = torch.tensor(dev_embedding[i:i + batch_size], dtype=torch.float, device=device)
                n_sents = torch.tensor(dev_n_sents[i:i + batch_size], dtype=torch.int, device=device)
                _, tmp_q = model(embeddings, n_sents)  # (b, n_class)
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)
                y_pred = tmp_q.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                ACC += cluster_acc(dev_target, y_pred)
                NMI += nmi(dev_target, y_pred)
                ARI += ari(dev_target, y_pred)
                AMI += ami(dev_target, y_pred)

        res = {'ACC' : ACC / step, 'NMI' : NMI / step, 'ARI' : ARI / step, 'AMI' : AMI / step}
        return res, delta_label
                
    
    ### pretrain AE
    model = IDEC(d_model = 768, n_emb = 256, n_cluster = n_class).to(device)
    assert isinstance(model, nn.Module)
    model.ae = pretrain_ae(args, model.ae, train_embedding, pretrain_path = args.output_path, patience = patience)
    
    ### cluster paramater initiate
    x = torch.tensor(train_embedding, dtype=torch.float, device=device)
    y = train_target
    x_prime, z = model.ae(x)
    
    y_pred, kmeans, acc, scores = clustering_single_trial(y, z.flatten(1), num_classes=n_class)
    
    x_prime = None
    z = None
    
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    
    ### training within an epoch
    optimizer = optim.SGD(model.parameters(), lr=lr)
    patience_count = 0
    
    model.train()
    for epoch in tqdm(range(n_epoch), desc='train'):
        epoch_loss = 0
        ### update p distribution
        _, tmp_q = model(x)
        tmp_q = tmp_q.data
        p = target_distribution(tmp_q)
        for i in range(0, len(train_embedding), batch_size):
            x = torch.tensor(train_embedding[i:i + batch_size], dtype=torch.float, device=device)
            n_sents = torch.tensor(train_n_sents[i:i+batch_size], dtype=torch.int, device=device)
            targets = torch.tensor(train_target[i:i + batch_size], device=device)
            x_prime, q = model(x)
            
            reconstr_loss = F.mse_loss(x_prime.squeeze(1), x)
            kl_loss = F.kl_div(q.log(), p[i:i+batch_size], reduction = 'mean')
            loss = args.gamma * kl_loss + reconstr_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("epoch {} loss = {:.4f}".format(epoch, epoch_loss / i))
        eval_metric, delta_label = eval(model, dev_embedding, dev_n_sents, dev_target, y_pred, batch_size)
        if epoch > 0 and delta_label > args.tol:
            print('delta_label {:.4f}'.format(delta_label), '< tol', args.tol)
            print('Reached tolerance threshold. Stopping training.')
            best_model = deepcopy(model).cpu()
            break
    return best_model, eval_metric

def dataset_training(dataset, args):
    logging.info('Training on {}'.format(dataset))

    if dataset in ['20ng', "reuters", "fake_news"]:
        max_num_seq = 32
    elif dataset == 'mr':
        max_num_seq = 64
    elif dataset == "corona":
        max_num_seq = 8
    else:
        raise NotImplementedError

    if os.path.exists(args.output_path+f"/{dataset}_embedding_bank.pt"):
        print(f"""embedding bank file exists, loading from {args.output_path+f"/{dataset}_embedding_bank.pt"}""")
        embedding_checkpoint = torch.load(args.output_path+f"/{dataset}_embedding_bank.pt")
        embedding,target,n_class = embedding_checkpoint["embedding"],embedding_checkpoint["target"], embedding_checkpoint["n_class"]

    else:
        print(f"preparing embedding bank for {dataset}")
        # embedding = np.ones((256, 32, 768))
        # target = np.ones((256,))
        # n_class = 8
        embedding, target, n_class = get_embedding_bank(dataset, args)
        print(f"""save to {args.output_path+f"/{dataset}_embedding_bank.pt"}""")
        torch.save({"embedding":embedding,
                    "target":target,
                    "n_class":n_class,
                    }, args.output_path+f"/{dataset}_embedding_bank.pt")
    n_sents = torch.tensor([len(e) for e in embedding])

    embedding = formatting(embedding, max_num_seq)

    logging.info(f"""
    -----------------------------------------------------------
                           {dataset}
    -----------------------------------------------------------
    """)
    print(f"""
    -----------------------------------------------------------
                           {dataset}
    -----------------------------------------------------------
    """)



    if args.do_prediction:
        print("classification")
        logging.info("classification")
        avg_metric = train_kfold_classifier(args, n_class, embedding, n_sents, target)
        for key, val in avg_metric.items():
            print(f"{key}::{val}")
            logging.info(f"{key}::{val}")


    if args.do_clustering:
        print("clustering")
        logging.info("clustering")
        _, eval_metric = train_cluster(n_class, embedding, n_sents, target)
        for key, val in eval_metric.items():
            print(f"{key}::{val}")
            logging.info(f"{key}::{val}")
    
    
    if args.do_idec:
        print("idec")
        logging.info("idec")
        _,eval_metric = train_kfold_idec(args, n_class, embedding, n_sents, target)
        for key, val in eval_metric.items():
            print(f"{key}::{val}")
            logging.info(f"{key}::{val}")
        
        
    print("done")





DATASETS=['20ng', 'reuters', 'mr', "fake_news", "corona"]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        default="bert-base-uncased",
                        # default="nli-bert-base",
                        # default="/raid1/p3/jiahao/simsiam/runs/my-sup-simcse-bert-base-uncased"
                        # default="/raid1/p3/jiahao/simsiam/runs/bert_nli_64seqlen/checkpoints.cp"
                        )

    parser.add_argument("--model_type",
                        default="plm",
                        # default="sbert",
                        # default="simcse",
                        # default="blendrst",
                        help="plm is origianl pretrained lm, sbert is sentence bert, simcse is Simcse, and blendrst is our method")

    parser.add_argument("--output_path",
                        default="/raid1/p3/jiahao/simsiam/runs/document_leve/bert-base-uncased-test",
                        # default="/raid1/p3/jiahao/simsiam/runs/document_leve/sbert-nli",
                        # default="/raid1/p3/jiahao/simsiam/runs/document_leve/sup-simcse-bert",
                        # default="/raid1/p3/jiahao/simsiam/runs/document_leve/blendrst-bert-nli"
                        )

    parser.add_argument("--max_seq_length", type=int, default=32)

    parser.add_argument("--is_debug", action="store_true")
    parser.add_argument("--n_fold", type=int, default=5)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--datasets", default="all", type=str)
    parser.add_argument("--do_clustering", action="store_true",
                        help="use this to run clustering")
    parser.add_argument("--do_prediction", action="store_true",
                        help="use this for prediction")
    parser.add_argument("--do_idec", action="store_true",
                        help="use this for prediction")
    parser.add_argument("--tol", type = float, default=0.001,
                        help="use this for IDEC early stopping")
    parser.add_argument('--gamma', default=0.1, type=float, help='coefficient of clustering loss')

    args = parser.parse_args()
    assert args.datasets in DATASETS+["all"]
    if args.datasets =="all":
        args.datasets = DATASETS
    else:
        args.datasets = [args.datasets]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    logging.basicConfig(filename=args.output_path + '/pooling_test.log', level=logging.INFO)
    import time
    logging.info('''-------------------------------------------------
    {}
    ------------------------------------------------'''.format(time.asctime(time.localtime(time.time()))))


    for dataset in args.datasets:
        dataset_training(dataset, args)

if __name__=="__main__":
    main()