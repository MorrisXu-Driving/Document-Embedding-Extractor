import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MLPProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPProjector, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                    nn.LayerNorm(output_dim))

        self.n_layers = 3

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


class MLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPPredictor, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.ReLU(inplace=True))

        self.layer2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class UpperTriangularMLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(UpperTriangularMLPPredictor, self).__init__()
        self.proj_matrix = nn.Parameter(torch.randn(1, input_dim, hidden_dim))  # [b, d_in, d_h]
        self.bias = nn.Parameter(torch.randn(1, hidden_dim))  # [b, d_h]
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.func = nn.ReLU()

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: [b, d_in]
        hidden = torch.matmul(x[:, None, :], torch.triu(self.proj_matrix)).squeeze(1)  # [b, d_h]
        hidden = self.func(self.layernorm(hidden + self.bias))
        return self.output_layer(hidden)


class LABELPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LABELPredictor, self).__init__()

        self.layer = nn.Sequential(nn.Linear(input_dim * 3, output_dim))

        # self.layer = nn.Sequential(nn.Linear(input_dim * 3, input_dim),
        #                            nn.LayerNorm(input_dim),
        #                            nn.ReLU(inplace=True),
        #                            nn.Linear(input_dim, output_dim))

    def forward(self, x):
        return self.layer(x)


class Backbone(nn.Module):
    def __init__(self, type='albert-base-v2'):
        super(Backbone, self).__init__()
        self.d_model = 768
        print(f'using {type} as the backbone')
        self.model = AutoModel.from_pretrained(type)

    def forward(self, x, seq_lens=None, is_norm=True):
        outputs = self.model(x).last_hidden_state  # [b,s,d]

        if seq_lens is not None:
            res = []
            for output, seq_len in zip(outputs, seq_lens):
                res.append(output[:seq_len, :].mean(dim=-2).unsqueeze(0))
            res = torch.cat(res, dim=0)
        else:
            res = outputs.mean(dim=-2)

        if is_norm:
            res = F.normalize(res, p=2, dim=-1)

        # res = outputs[:, 0, :]
        return res


class Simsiam(nn.Module):
    def __init__(self, backbone, projector, predictor, mlm_cls, label_cls):
        super(Simsiam, self).__init__()
        self.backbone = backbone
        self.projector = projector
        self.encoder = nn.Sequential(self.backbone, self.projector)
        self.mlm_cls = mlm_cls
        self.label_cls = label_cls
        self.predictor = predictor

        for p in nn.Sequential(projector, predictor).parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)

    def forward(self, x1, x2, seq_len):
        b1, b2 = self.backbone(x1, seq_len), self.backbone(x2, seq_len)
        z1, z2 = self.projector(b1), self.projector(b2)
        p1, p2 = self.predictor(z1), self.predictor(z2)

        return z1, z2, p1, p2
