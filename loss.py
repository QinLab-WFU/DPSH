import torch
import torch.nn.functional as F
from torch import nn

from miner import DistanceWeightedMiner


class MarginLoss(nn.Module):
    def __init__(self, n_bits, n_classes, **kwargs):
        super().__init__()

        self.alpha = kwargs.pop("alpha", 0.2)
        self.beta = nn.Parameter(torch.ones(n_classes) * kwargs.pop("beta", 1.2))

        self.miner = DistanceWeightedMiner(n_bits, **kwargs)

    def forward(self, batch, labels):
        dist_mat = torch.cdist(batch, batch)

        anc_idxes, pos_idxes, neg_idxes = self.miner(dist_mat.detach(), labels)

        d_ap = dist_mat[anc_idxes, pos_idxes]
        d_an = dist_mat[anc_idxes, neg_idxes]

        anchor_labels = labels[anc_idxes]
        beta = torch.einsum("nc,c->n", anchor_labels, self.beta) / anchor_labels.sum(dim=1)

        pos_loss = F.relu(d_ap - beta + self.alpha)
        neg_loss = F.relu(beta - d_an + self.alpha)

        pair_count = torch.sum((pos_loss > 0.0) + (neg_loss > 0.0))

        loss = torch.sum(pos_loss + neg_loss) if pair_count == 0.0 else torch.sum(pos_loss + neg_loss) / pair_count

        return loss
