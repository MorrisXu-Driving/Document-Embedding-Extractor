import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class RetMetric(object):
    def __init__(self, feats, labels):
        self.gallery_feats = self.query_feats = feats
        self.gallery_labels = self.query_labels = labels
        self.sim_mat = np.matmul(self.query_feats, np.transpose(self.gallery_feats))

    def recall_k(self, K=[1]):
        m = len(self.sim_mat)

        match_counter = [0] * len(K)

        for i in range(m):
            pos_sim = self.sim_mat[i][self.gallery_labels == self.query_labels[i]]
            neg_sim = self.sim_mat[i][self.gallery_labels != self.query_labels[i]]

            thresh = np.sort(pos_sim)[-2] if pos_sim.size > 1 else np.max(pos_sim)
            n_violation = np.sum(neg_sim > thresh)
            for j, k in enumerate(K):
                if n_violation < k:
                    match_counter[j] += 1
        return np.asarray(match_counter) / m
    
    
def visulizing_embeddings(embedding_bank, label_bank, dataset, output_path, fold = None):
    n_classes = label_bank.max() + 1
    colors = cm.rainbow(label_bank.astype(np.float32) / n_classes)[:, 0]
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    transferred_embeddings = tsne.fit_transform(embedding_bank)

    plt.scatter(transferred_embeddings[:, 0], transferred_embeddings[:, 1], c=colors)

    if not os.path.exists(os.path.join(output_path,'visual', dataset)):
        os.makedirs(os.path.join(output_path, 'visual', dataset))
    plt.title(f't-SNE Visualization of Embedding Space on {dataset}')
    plt.savefig(os.path.join(output_path, 'visual', dataset, str(fold) if fold is not None else 'embedding' + '.png' ))


def eval_cnn(cls, embedding_bank, target_bank, batch_size=1, max_num_seq=64, dataset=None, fold=0, K=[1], device=0, output_path='./'):
    assert isinstance(cls, nn.Module)
    cls = cls.to(torch.device(f'cuda:{device}'))
    cls = cls.eval()
    
    for p in cls.parameters():
        p.requires_grad = False
    predicted_bank = []
    with torch.no_grad():
        for i in tqdm(range(0, len(embedding_bank), batch_size), desc='eval'):
            embeddings = embedding_bank[i:i + batch_size]
            input_tensor = torch.tensor(embeddings, dtype=torch.float).to(device)
            output_tensor = cls(input_tensor)  # (b, emb_dim)
            assert isinstance(output_tensor, torch.Tensor)
            predicted_bank.append(output_tensor.data.cpu().numpy())
            if i + 1 % 100 == 0:
                logging.info(f'Extract Features: [{i + 1}/{int(len(embedding_bank) / batch_size)}]')
            del output_tensor
        predicted_bank = np.vstack(predicted_bank)
    metric = RetMetric(predicted_bank, target_bank)
    visulizing_embeddings(predicted_bank, target_bank, dataset, output_path, fold)
    return metric.recall_k(K)