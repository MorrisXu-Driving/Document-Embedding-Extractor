import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment as linear_assignment

def get_lr(optimizer):
    for p in optimizer.param_groups:
        return p["lr"]

def visulizing_embeddings(embedding_bank, label_bank, dataset, fold, output_path):
    n_classes = label_bank.max() + 1
    colors = cm.rainbow(label_bank.astype(np.float32) / n_classes)[:, 0]
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    transferred_embeddings = tsne.fit_transform(embedding_bank)
    plt.scatter(transferred_embeddings[:, 0], transferred_embeddings[:, 1], c=colors)
    if not os.path.exists(os.path.join(output_path,'visual', dataset)):
        os.makedirs(os.path.join(output_path, 'visual', dataset), exist_ok=True)
    plt.savefig(os.path.join(os.getcwd(), 'visual', dataset, str(fold) if fold is not None else 'embedding' + '.pdf' )
                , dpi=400, bbox_inches='tight', pad_inches=0)

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
    
    
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
