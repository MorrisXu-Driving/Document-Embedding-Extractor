import torch
import torch.nn as nn

class CNNPoolingClassifier(nn.Module):
    def __init__(self, d_model, emb_dim, pooling_ops):
        super(CNNPoolingClassifier, self).__init__()
        c = d_model // 3
        self.cnn1 = nn.Conv2d(1, c, (2, d_model))
        self.cnn2 = nn.Conv2d(1, c, (3, d_model))
        self.cnn3 = nn.Conv2d(1, c, (4, d_model))
        
        if pooling_ops == 'max':
            self.pooling = lambda x: torch.max(x, dim=-1)[0]
        elif pooling_ops == 'max_mean':
            self.pooling = lambda x: torch.max(x, dim=-1)[0] + torch.mean(x, dim = -1)

        self.cls = nn.Sequential(nn.Linear(d_model, d_model),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(d_model, d_model),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(d_model, emb_dim))

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