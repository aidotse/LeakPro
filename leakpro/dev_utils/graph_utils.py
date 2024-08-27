"""Functions relating to computational graph investigations."""
import torch


def calculate_graph_depth(grad_fn: torch.nn.Module, cache: dict=None) -> int:
    """Calculates graph depth from a tensors grad_fn."""
    if not grad_fn:
        return 0
    if cache is None:
        cache = {}

    if grad_fn in cache:
        return cache[grad_fn]

    max_depth = 0
    for sub_fn, _ in grad_fn.next_functions:
        if sub_fn is not None:
            sub_depth = calculate_graph_depth(sub_fn, cache) + 1
            max_depth = max(sub_depth, max_depth)

    cache[grad_fn] = max_depth
    return max_depth
