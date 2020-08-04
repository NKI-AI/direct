# coding=utf-8
# Copyright (c) DIRECT Contributors

# From:
# https://github.com/pytorch/pytorch/blob/00aa23446b9d2b3dac5ed8b343c4536f7d9dd8df/torch/utils/data/_utils/collate.py#L42
import torch

from torch.utils.data._utils.collate import (
    default_convert,
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
)
from torch._six import container_abcs, string_classes, int_classes


def named_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size.
    If named tensor adds batch to the first dimension."""
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        out = None
        # TODO: Named tensor: once stack supports named tensors, drop the rename.
        names = elem.names
        elem = elem.rename(None)
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.rename(None).numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        out_batch = torch.stack([_.rename(None) for _ in batch], 0, out=out)
        if any(names):
            new_names = tuple(["batch"] + list(names))
            out_batch = out_batch.refine_names(*new_names)
        return out_batch

    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return named_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: named_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(named_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [named_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
