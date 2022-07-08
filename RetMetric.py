import numpy as np

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