import torch
from typing import Optional, Union
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter


def tailfreeSample(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    temperature=0.5
) -> Tensor:
    softmax = nn.Softmax(dim=-1)

    x = x/float(temperature)

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

    # 获取每个图中节点概率最小的值
    min_values = probs.gather(1, num_nodes.view(-1, 1)-1)
    probs = torch.where(probs > 0, probs, min_values)

    grad = probs[:, 1:] - probs[:, :-1]
    grad = grad[:, 1:] - grad[:, :-1]

    only_pos = torch.abs(grad)
    sum = torch.sum(only_pos, dim=1).view(-1, 1)
    sec_weights = only_pos/sum
    cum_weights = (torch.cumsum(sec_weights, dim=1) > 0.9)+0

    tail_ids = torch.argmax(cum_weights, dim=1)+1
    k = (float(ratio) * tail_ids).ceil().to(torch.int)
    k = torch.clamp(tail_ids, min=1)

    index = torch.cat([
        torch.arange(k[i], device=x.device) + i * max_num_nodes
        for i in range(batch_size)
    ], dim=0)

    perm = perm[index]

    return perm
