import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union
from torch_geometric.utils import scatter
import torch.nn as nn


def pre_select(
        score: Tensor,
        num_nodes: int,
        ratio=0.5
) -> Tensor:
    softmax = nn.Softmax(dim=-1)
    if num_nodes > 1:
        score = score[:num_nodes-1]
    sm = softmax(score)
    sm, perm = sm.sort(dim=0)

    cdf = torch.cumsum(sm, dim=-1)
    normalized_cdf = (
        cdf - cdf.min(dim=0)[0]
    ) / ((cdf.max(dim=0)[0] - cdf.min(dim=0)[0]) / 1.0)

    T = int(ratio*num_nodes) if int(ratio*num_nodes) > 0 else 1
    ys = torch.linspace(
        start=0,
        end=1.0,
        steps=T+2,
        device=score.device
    )[1:T+1].view(-1, 1)

    pre_cdf = normalized_cdf.repeat(T, 1)
    pre_selected_ind = torch.argmax((pre_cdf > ys)+0, dim=1)
    perm = perm[pre_selected_ind].unique()
    return perm


def inverse_transform_sampling(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
) -> Tensor:

    perm = torch.arange(x.size(0), device=x.device)

    num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

    batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes, ), -60000.0)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    m = torch.cat([
        pre_select(score=dense_x[i], num_nodes=num_nodes[i],
                   ratio=ratio) + cum_num_nodes[i]
        for i in range(batch_size)
    ], dim=0)

    perm = perm[m]

    return perm
