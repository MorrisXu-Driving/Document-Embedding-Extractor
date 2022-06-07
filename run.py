import os
import sys
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch
import transformers
from transformers import AutoModel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold

from utils import *
from model import CNNPoolingClassifier
from evaluate import eval_cnn 
from trainer import train_cnn

def document_embedding(model, doc, tokenizer, embedding_batch_size=64):
    embedding = []
    output_features = tokenizer(doc, padding='max_length', truncation=True, return_tensors='pt')
    n_sents = len(output_features.data['input_ids'])
    for i in range(0, n_sents, embedding_batch_size):
        input_ids = output_features.data['input_ids'][i:i + embedding_batch_size]
        length = output_features.data['attention_mask'][i:i + embedding_batch_size].sum(dim=-1)
        sent_embeddings = model(input_ids.to(device), length).cpu()
        embedding.append(sent_embeddings)
    embedding = torch.cat(embedding, dim=0)
    return embedding.numpy()


def get_sent_embedding(docs):
    embedding_bank = []
    # pretrained sentence bert
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(backbone)
    model = model.eval()
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        for doc in tqdm(docs, desc='embedding'):
            embedding_bank.append(model.encode(doc, show_progress_bar=False))
    return embedding_bank


def get_embedding_bank(dataset, max_num_seq):
    if dataset == '20ng':
        input, target = prepare_20ng()
    elif dataset == 'reuters':
        input, target = prepare_reuters()
    elif dataset == 'mr':
        input, target = prepare_movie_review()
    else:
        raise NotImplementedError

    emb_dim = 256
    if is_debug:
        input, target = input[:100], target[:100]
    embedding = get_sent_embedding(input)
    embedding = formatting(embedding, max_num_seq)  # (N, n_sentences, word_embedding_dim)
    return embedding, np.array(target), emb_dim


def dataset_training(dataset):
    logging.info('Training on {}'.format(dataset))
    if dataset == '20ng':
        max_num_seq = 32
    elif dataset == 'reuters':
        max_num_seq = 32
    else:
        raise NotImplementedError
    
    K = [1, 2, 4, 8, 16]
    if not is_debug:
        embedding, target, emb_dim = get_embedding_bank(dataset, max_num_seq)
    else:
        embedding = np.ones((256, 32, 768))
        target = np.ones((256,))
        emb_dim = 256

    logging.info('{} map to embeddings with {} dim'.format(dataset, emb_dim))
    kf = KFold(n_fold)
    for fold, (train_index, dev_index) in enumerate(kf.split(embedding)):
        train_embedding, train_target = embedding[train_index], target[train_index]
        dev_embedding, dev_target = embedding[dev_index], target[dev_index]
        cnn_classifier = CNNPoolingClassifier(768, emb_dim, pooling_ops)
        cnn_classifier = train_cnn(cnn_classifier, train_embedding, train_target, n_epoch, batch_size, lr, patience)
        recall_fold = eval_cnn(cnn_classifier, dev_embedding, dev_target, batch_size, max_num_seq, dataset, fold, K = K, output_path = output_path)
        if fold == 0:
            recall_sum = recall_fold
        else:
            recall_sum += recall_fold
            
        fold_res = f'Fold{fold}::' + ' '.join([f'Recall@{k}:{v:.4f}' for k, v in zip(K, recall_fold)])
        logging.info(fold_res)
        print(fold_res)
        
    recall_sum /= n_fold
    overall_res = 'Overall::' + ' '.join([f'Recall@{k}:{v:.4f}' for k, v in zip(K, recall_sum)])
    logging.info(overall_res)
    print(overall_res)


def main():
    for task in tasks:
        dataset_training(task)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'arguments for training and evaluating the document pooler')
    parser.add_argument('--backbone', type = str, default = 'nli-bert-base', help = 'backbone for extracting sentence embedding from the original text, can be either nli-bert-base or stsb-bert-base')
    parser.add_argument('--pooling_ops', type = str, default = 'max', help = 'the pooling operation in the model')
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--n_epoch', type = int, default = 200)
    parser.add_argument('--n_fold', type = int, default = 5)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--margin', type = float, default = 0.05, help = 'margin in the metric learning objective')
    parser.add_argument('--is_debug', type = bool, default = False, help = 'whether in the debug mode')
    parser.add_argument('--device', type = int, default = 0, help = 'index of the gpu to be used')
    args = parser.parse_args()
    
    backbone = args.backbone
    pooling_ops = args.pooling_ops
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    n_fold = args.n_fold
    lr = args.lr
    margin = args.margin
    device = args.device
    is_debug = args.is_debug
    
    
    output_path = './logs/cnn/clustering/{}'.format(backbone)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(filename = output_path + '/train.log', level = logging.INFO)
    import time
    logging.info('''-------------------------------------------------
    {}
    ------------------------------------------------'''.format(time.asctime(time.localtime(time.time()))))
    
    logging.info('''
    batch_size = {} lr = {} margin = {} optimizer = SGD
    ------------------------------------------------'''.format(batch_size, lr, margin))
    patience = 5
    tasks = ['20ng', 'reuters']
    main()
