import torch
from typing import Optional, Union
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter


def nucleusSample(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
) -> Tensor:
    softmax = nn.Softmax(dim=-1)

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
    dense_x = softmax(dense_x)

    probs, perm = dense_x.sort(dim=-1, descending=True)

    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    cum_probs = torch.cumsum(probs, dim=-1)
    nucleus = (cum_probs < ratio)+0

    if ratio >= 1:
        k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
        k = torch.min(k, num_nodes)
    else:
        k = torch.count_nonzero(nucleus, dim=1).reshape(-1)
        k = torch.clamp(k, min=1)

    if isinstance(ratio, int) and (k == ratio).all():
        # If all graphs have exactly `ratio` or more than `ratio` entries,
        # we can just pick the first entries in `perm` batch-wise:
        index = torch.arange(batch_size, device=x.device) * max_num_nodes
        index = index.view(-1, 1).repeat(1, ratio).view(-1)
        index += torch.arange(ratio, device=x.device).repeat(batch_size)
    else:
        # Otherwise, compute indices per graph:
        index = torch.cat([
            torch.arange(k[i], device=x.device) + i * max_num_nodes
            for i in range(batch_size)
        ], dim=0)

    perm = perm[index]

    return perm
