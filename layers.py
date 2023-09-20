from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
import torch

from nucleus import nucleusSample
from tailfree import tailfreeSample
from inverse import inverse_transform_sampling as inverse


class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio, sampling_method, Conv=GCNConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity
        self.sampling_method = sampling_method

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x, edge_index).squeeze()

        if self.sampling_method == 'TOPK':
            perm = topk(score, self.ratio, batch)
        elif self.sampling_method == 'NUCLEUS':
            perm = nucleusSample(score, self.ratio, batch)
        elif self.sampling_method == 'TAILFREE':
            perm = tailfreeSample(score, self.ratio, batch)
        elif self.sampling_method == 'ITS':
            perm = inverse(score, self.ratio, batch)

        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
