"""
Classes defining the generation context (conditioning image and camera intrinsics matrix) and
a training example (point cloud and context). Additional utils for pretty-printing etc.
"""
from typing import NamedTuple

import torch
from torch import Tensor


def _raw_repr(obj) -> list[str]:
    """
    A helper for implementing __repr__. It returns a list of lines which
    describe the object. Works recursively for objects which have a _enumerate_fields.
    The reason for returning a list of lines is to enable indented printing of
    nested objects.
    """
    lines = []
    lines.append(f"{type(obj).__name__}(")

    for name, value in obj._enumerate_fields():
        if hasattr(value, "_raw_repr"):
            head, *tail, end = value._raw_repr()
            lines.append(f" {name}={head}")
            for line in tail:
                lines.append(f"  {line}")
            lines.append(f" {end}")
        elif torch.is_tensor(value):
            lines.append(f" {name}={tuple(value.shape)},")
        else:
            lines.append(f" {name}={value},")

    lines.append(f")")
    return lines


def apply_to_tensors(obj: object, f: callable) -> object:
    """
    Applies a function `f` to all tensors in the object. Works out-of-place
    """
    applied = {}
    for name, value in obj._enumerate_fields():
        if hasattr(value, "apply_to_tensors"):
            applied[name] = value.apply_to_tensors(f)
        elif torch.is_tensor(value):
            applied[name] = f(value)
        else:
            applied[name] = value

    return type(obj)(**applied)


class DataError(RuntimeError):
    pass


def _named_tuple_enumerate_fields(obj: NamedTuple):
    yield from obj._asdict().items()


# class Context3d(NamedTuple):
#     """
#     A class representing the context of a generation. It consists of a conditioning
#     image and a camera matrix. The camera intrisics matrix is 3x3.
#     """

#     image: Tensor
#     K: Tensor
#     gaussian_pcd: Tensor
#     boneflow: Tensor
#     skinning: Tensor
#     joints_mask: Tensor
#     shape_name: Tensor

#     _enumerate_fields = _named_tuple_enumerate_fields
#     _raw_repr = _raw_repr
#     apply_to_tensors = apply_to_tensors

#     def __repr__(self) -> str:
#         return "\n".join(self._raw_repr())


class Context3d(NamedTuple):
    """
    A class representing the context of a generation. It consists of a conditioning
    image and a camera matrix. The camera intrisics matrix is 3x3.
    """

    image: Tensor
    K: Tensor
    gaussian_pcd: Tensor
    shape_name: Tensor
    # boneflow: Tensor
    joints_mask: Tensor

    _enumerate_fields = _named_tuple_enumerate_fields
    _raw_repr = _raw_repr
    apply_to_tensors = apply_to_tensors

    def __repr__(self) -> str:
        return "\n".join(self._raw_repr())    
    

class Example(NamedTuple):
    """
    A class representing a training example. It consists of a point cloud and a context (possibly None).
    """

    data: Tensor | None
    ctx: Context3d | None

    _enumerate_fields = _named_tuple_enumerate_fields
    _raw_repr = _raw_repr
    apply_to_tensors = apply_to_tensors

    def __repr__(self) -> str:
        return "\n".join(self._raw_repr())
