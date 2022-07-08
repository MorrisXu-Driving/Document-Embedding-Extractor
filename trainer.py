from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses
from pytorch_metric_learning.miners import BatchHardMiner, MultiSimilarityMiner


def train_cnn(cls, embedding_bank, target_bank, n_epoch, batch_size=1, lr=1e-3, patience=5, device=0):
    assert isinstance(cls, nn.Module)
    cls = cls.train()
    cls = cls.to(torch.device(f'cuda:{device}'))

    optimizer = optim.SGD(cls.parameters(), lr=lr)
    criterion = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
    miner = MultiSimilarityMiner(epsilon=0.1)
    
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
            output_tensor = cls(input_tensor)  # (b, emb_dim)
            target_tensor = torch.tensor(targets).to(device)
            loss = criterion(output_tensor, target_tensor, miner(output_tensor, target_tensor))

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