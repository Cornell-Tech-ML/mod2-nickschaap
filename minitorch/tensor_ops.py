from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce placeholder"""
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            start: starting value for reduction

        Returns:
        -------
            :class:`TensorData` : new tensor

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Compute the broadcasted shape
        try:
            broadcasted_shape = shape_broadcast(tuple(out_shape), tuple(in_shape))
        except ValueError as e:
            raise ValueError(
                f"Cannot broadcast shapes {out_shape} and {in_shape}"
            ) from e

        # Total number of elements in the broadcasted tensor
        total_elements = int(operators.prod(broadcasted_shape))

        # Precompute the dimension difference for broadcasting
        out_dim_diff = len(broadcasted_shape) - len(out_shape)
        in_dim_diff = len(broadcasted_shape) - len(in_shape)

        # Iterate over each element in the broadcasted tensor
        for idx in range(total_elements):
            # Compute the multi-dimensional index for the broadcasted tensor
            multi_index = np.array([0] * len(broadcasted_shape))
            to_index(idx, np.array(broadcasted_shape), multi_index)

            # Map to output tensor index
            out_index = multi_index[out_dim_diff:]
            out_flat_index = index_to_position(out_index, out_strides)

            # Map to input tensor index using broadcasting
            in_index = multi_index[in_dim_diff:]
            broadcast_index(in_index, np.array(broadcasted_shape), in_shape, in_index)
            in_flat_index = index_to_position(in_index, in_strides)

            # Apply the function and store the result
            out[out_flat_index] = fn(in_storage[in_flat_index])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
    ----
        fn: function mapping two floats to float to apply

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Compute the total number of elements in the output tensor.
        out_size = 1
        for s in out_shape:
            out_size *= s

        max_dims = len(out_shape)

        # Function to pad shapes and strides to the maximum dimensions.
        def pad_list(lst: np.ndarray, max_dims: int, pad_value: int) -> np.ndarray:
            return np.array([pad_value] * (max_dims - len(lst)) + list(lst))

        # Pad shapes and strides for broadcasting.
        a_shape_padded = pad_list(a_shape, max_dims, 1)
        a_strides_padded = pad_list(a_strides, max_dims, 0)
        b_shape_padded = pad_list(b_shape, max_dims, 1)
        b_strides_padded = pad_list(b_strides, max_dims, 0)

        for i in range(out_size):
            # Convert the flat index to a multidimensional index for the output tensor.
            out_index = np.array([0] * max_dims)
            to_index(i, out_shape, out_index)

            # Adjust indices for tensors a and b based on broadcasting rules.
            a_index = np.array(
                [
                    0 if a_shape_padded[dim] == 1 else out_index[dim]
                    for dim in range(max_dims)
                ]
            )
            b_index = np.array(
                [
                    0 if b_shape_padded[dim] == 1 else out_index[dim]
                    for dim in range(max_dims)
                ]
            )

            # Calculate flat positions using strides.
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides_padded)
            b_pos = index_to_position(b_index, b_strides_padded)

            # Apply the function and store the result.
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # Compute the total number of elements in the output tensor.
        out_size = 1
        for s in out_shape:
            out_size *= s

        # Iterate over each element in the output tensor.
        for i in range(out_size):
            # Convert the flat index to a multidimensional index for the output tensor.
            out_index = np.array([0] * len(out_shape))
            to_index(i, out_shape, out_index)

            # Prepare an index for the input tensor 'a'.
            a_index = out_index.copy()

            # Initialize the reduction result with the first element along the reduction dimension.
            a_index[reduce_dim] = 0
            a_pos = index_to_position(a_index, a_strides)
            result = a_storage[a_pos]

            # Iterate over the reduction dimension.
            for rd in range(1, a_shape[reduce_dim]):
                a_index[reduce_dim] = rd
                a_pos = index_to_position(a_index, a_strides)
                result = fn(result, a_storage[a_pos])

            # Store the result in the output storage.
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = result

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
