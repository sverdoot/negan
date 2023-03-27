from typing import Sequence, Union

import numpy as np
import torch
from einops import rearrange


def prepare_matrix(
    kernel: torch.Tensor,
    pad_to: Sequence[int],
    strides: Union[int, Sequence[int]],
):
    if len(kernel.shape) not in (4, 5):
        raise RuntimeError(kernel.shape)
    if len(pad_to) != len(kernel.shape) - 2:
        raise RuntimeError((pad_to, kernel.shape))
    dim = len(pad_to)
    if isinstance(strides, int):
        strides = [strides] * dim
    else:
        if len(strides) != dim:
            raise RuntimeError()
    for i in range(dim):
        assert pad_to[i] % strides[i] == 0
        assert kernel.shape[i] <= pad_to[i], (kernel.shape, pad_to)
    old_shape = kernel.shape
    kernel_tr = torch.permute(kernel, dims=[dim, dim + 1] + list(range(dim)))
    padding_tuple = []
    for i in range(dim):
        padding_tuple.append(0)
        padding_tuple.append(pad_to[-i - 1] - kernel_tr.shape[-i - 1])
    kernel_pad = torch.nn.functional.pad(kernel_tr, tuple(padding_tuple))
    r1, r2 = kernel_pad.shape[:2]
    small_shape = []
    for i in range(dim):
        small_shape.append(pad_to[i] // strides[i])
    kernel_for_fft = torch.zeros(
        (r1, r2, np.prod(np.array(strides))) + tuple(small_shape),
    )
    if dim == 2:
        p1, p2 = pad_to[0], pad_to[1]
        kernel_for_fft = kernel_pad.reshape(r1, r2, p1, p2 // strides[1], strides[1])
        kernel_for_fft = kernel_for_fft.reshape(
            r1,
            r2,
            p1 // strides[0],
            strides[0],
            p2 // strides[1],
            strides[1],
        )
        kernel_for_fft = kernel_for_fft.permute(0, 1, 5, 3, 4, 2)
        kernel_for_fft = kernel_for_fft.reshape(
            r1,
            r2,
            np.prod(strides),
            p1 // strides[0],
            p2 // strides[1],
        )
        kernel_for_fft = kernel_for_fft.transpose(4, 3)

        # for i in range(strides[0]):
        #     for j in range(strides[1]):
        #         kernel_for_fft[:, :, i * strides[1] + j, ...] = kernel_pad[...,
        # i :: strides[0], j :: strides[1]]
    else:
        for i in range(strides[0]):
            for j in range(strides[1]):
                for k in range(strides[2]):
                    index = i * strides[1] * strides[2] + j * strides[2] + k
                    kernel_for_fft[:, :, index, ...] = kernel_pad[
                        ..., i :: strides[0], j :: strides[1], k :: strides[2]
                    ]
    fft_results = torch.fft.fft2(kernel_for_fft).reshape(r1, -1, *small_shape)
    # sing_vals shape is (r1, 4r2, k, k, k)
    transpose_for_svd = fft_results.permute(list(range(2, dim + 2)) + [0, 1])
    # transpose_for_svd = np.transpose(fft_results, axes=list(range(2, dim + 2)) + \
    # [0, 1])
    # now the shape is (k, k, k, r1, 4r2)
    return kernel_pad, old_shape, r1, r2, small_shape, strides, transpose_for_svd


def get_sing_vals(
    kernel: torch.Tensor,
    pad_to: Sequence[int],
    stride: Union[int, Sequence[int]],
):
    kernel = kernel.cpu().permute([2, 3, 0, 1])
    if kernel.shape[0] > pad_to[0]:
        k, n = kernel.shape[0], pad_to[0]
        if not (k == n + 2 or k == n + 1):
            raise RuntimeError("?")
        pad_kernel = torch.nn.functional.pad(
            kernel,
            (0, 0, 0, 0, 0, max(k, 2 * n) - k, 0, max(k, 2 * n) - k),
        )
        tmp = rearrange(
            pad_kernel,
            "(w1 k1) (w2 k2) cin cout -> (k1 k2) (w1 w2) cin cout",
            w1=2,
            w2=2,
        )
        sv = torch.sqrt((tmp.sum(1) ** 2).sum(0))
        return sv

    before_svd = prepare_matrix(kernel, pad_to, stride)
    svdvals = torch.linalg.svdvals(before_svd[-1])
    return svdvals


# def get_sing_vals(kernel, pad_to, stride, periodic=True):
#     kernel = kernel.cpu().permute([2, 3, 0, 1])
#     if kernel.shape[0] > pad_to[0]:
#         k, n = kernel.shape[0], pad_to[0]
#         assert k == n + 1
#         if periodic:
#             pad_kernel = kernel.detach().clone()[:-1, :-1]
#             pad_kernel[0, 0] += kernel[-1, -1]
#             pad_kernel[0, :] += kernel[-1, :-1]
#             pad_kernel[:, 0] += kernel[:-1, -1]
#             kernel = pad_kernel
#         else:
#             out_ch, in_ch = kernel.shape[2:]
#             matrix = torch.zeros((out_ch * 4, in_ch * n * n))
#             for i in range(out_ch):
#                 matrix[i * 4] = kernel[1:, 1:, i].permute([2, 0, 1]).flatten()
#                 matrix[i * 4 + 1] = kernel[1:, :-1, i].permute([2, 0, 1]).flatten()
#                 matrix[i * 4 + 2] = kernel[:-1, 1:, i].permute([2, 0, 1]).flatten()
#                 matrix[i * 4 + 3] = kernel[:-1, :-1, i].permute([2, 0, 1]).flatten()
#             svdvals = torch.linalg.svdvals(matrix)
#             return svdvals
#     else:
#         assert periodic
#     before_svd = get_ready_for_svd(kernel, pad_to, stride)
#     svdvals = torch.linalg.svdvals(before_svd[-1])
#     return svdvals


def get_sing_vals_simple(kernel, pad_to, stride):
    svdvals = torch.linalg.svdvals(kernel)
    return svdvals
