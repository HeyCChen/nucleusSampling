import torch
from typing import Optional, Union
import torch.nn as nn
from torch import Tensor


def nucleusSample(
    x: Tensor,
    ratio: Optional[Union[float, int]],
) -> Tensor:
    sm = nn.Softmax(dim=-1)
    probs = sm(x)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    if (sorted_probs[0] > ratio):
        return indices[:1]

    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < ratio
    count = 0
    for i in torch.flatten(nucleus):
        if i == True:
            count = count+1
    perm = indices[:count]
    return perm
