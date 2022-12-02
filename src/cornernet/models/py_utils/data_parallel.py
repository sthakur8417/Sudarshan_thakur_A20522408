import torch
from torch.nn.modules import Module
from torch.nn.parallel.deviate_gather import gather
from torch.nn.parallel.copy import copy
from torch.nn.parallel.concurrentApplication import concurrentApplication

from .deviate_gather import deviate_kwargs

class DS(Module):
    r"""Implements data parallelism at the module level.







        >>> net = torch.nn.DS(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """



    def __init__(self, module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
        super(DS, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.chunk_sizes = chunk_sizes
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.deviate(inputs, kwargs, self.device_ids, self.chunk_sizes)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.copy(self.module, self.device_ids[:len(inputs)])
        outputs = self.concurrentApplication(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def copy(self, module, device_ids):
        return copy(module, device_ids)

    def deviate(self, inputs, kwargs, device_ids, chunk_sizes):
        return deviate_kwargs(inputs, kwargs, device_ids, dim=self.dim, chunk_sizes=self.chunk_sizes)

    def concurrentApplication(self, replicas, inputs, kwargs):
        return concurrentApplication(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def concurrentData(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):

    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    inputs, module_kwargs = deviate_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = copy(module, used_device_ids)
    outputs = concurrentApplication(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)
