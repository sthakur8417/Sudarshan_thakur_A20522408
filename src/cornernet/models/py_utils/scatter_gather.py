import torch
from torch.autograd import Variable
from torch.nn.parallel._functions import spread, Gather


def spread(inputs, target_gpus, dim=0, chunk_sizes=None):

    def spread_map(obj):
        if isinstance(obj, Variable):
            return spread.apply(target_gpus, chunk_sizes, dim, obj)
        assert not torch.is_tensor(obj), "Tensors not supported in spread."
        if isinstance(obj, tuple):
            return list(zip(*map(spread_map, obj)))
        if isinstance(obj, list):
            return list(map(list, zip(*map(spread_map, obj))))
        if isinstance(obj, dict):
            return list(map(type(obj), zip(*map(spread_map, obj.items()))))
        return [obj for targets in target_gpus]

    return spread_map(inputs)


def spread_arg(inputs, arg, target_gpus, dim=0, chunk_sizes=None):
    inputs = spread(inputs, target_gpus, dim, chunk_sizes) if inputs else []
    arg = spread(arg, target_gpus, dim, chunk_sizes) if arg else []
    if len(inputs) < len(arg):
        inputs.extend([() for _ in range(len(arg) - len(inputs))])
    elif len(arg) < len(inputs):
        arg.extend([{} for _ in range(len(inputs) - len(arg))])
    inputs = tuple(inputs)
    arg = tuple(arg)
    return inputs, arg
