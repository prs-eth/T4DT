import torch
import tntorch as tn
import numpy as np
import itertools
import operator
from typing import Callable, List, Optional
from functools import partial


def reduce(
        ts: List[tn.Tensor],
        round_f: Callable,
        function: Callable = partial(tn.cat, dim=-1),
        eps: float = 1e-14,
        rank: Optional[int] = None,
        algorithm: str = 'svd'):
    d = dict()
    for i, elem in enumerate(ts):
        climb = 0  # For going up the tree
        while climb in d:
            elem = round_f(function(d[climb], elem), rmax=rank, eps=eps, algorithm=algorithm)
            d.pop(climb)
            climb += 1
        d[climb] = elem
    keys = list(d.keys())
    result = d[keys[0]]
    for key in keys[1:]:
        result = round_f(function(result, d[key]), rmax=rank, eps=eps, algorithm=algorithm)
    return result


def reduce_tucker(
        ts: List[tn.Tensor],
        function: Callable = partial(tn.cat, dim=-1),
        eps: float = 1e-14,
        rank: Optional[int] = None,
        algorithm: str = 'svd'):
    return reduce(ts, tn.round_tucker, function, eps, rank, algorithm)


def reduce_tt(
        ts: List[tn.Tensor],
        function: Callable = partial(tn.cat, dim=-1),
        eps: float = 1e-14,
        rank: Optional[int] = None,
        algorithm: str = 'svd'):
    return reduce(ts, tn.round_tt, function, eps, rank, algorithm)


def is_pow2(n: int) -> bool:
    return n != 0 and ((n & (n - 1)) == 0)


def tensor3d2qtt(t: torch.Tensor, checks: bool = True) -> torch.Tensor:
    shape = t.shape
    if checks:
        assert len(shape) == 3
        assert shape[0] == shape[1] == shape[2], f'Only tensors with all equal dimensions are supported'
        assert is_pow2(shape[0])
    dim_grid = int(np.log(shape[0]) / np.log(2))
    num_dims = 3 * dim_grid
    qtt = t.reshape([2] * num_dims)
    dimentions = np.arange(num_dims)
    dims = list(
        itertools.chain.from_iterable(
            zip(
                dimentions[:dim_grid],
                dimentions[dim_grid:2 * dim_grid],
                dimentions[2 * dim_grid:])))
    qtt = qtt.permute(dims)
    return qtt


def tensor3d2oqtt(t: torch.Tensor, checks: bool = True) -> torch.Tensor:
    oqtt = tensor3d2qtt(t, checks)
    oqtt = oqtt.reshape([8] * len(oqtt.shape) // 3)
    return oqtt


def oqtt2tensor3d(oqtt: torch.Tensor, checks: bool = True) -> torch.Tensor:
    qtt = oqtt.reshape([2] * len(oqtt.shape) * 3)
    return qtt2tensor3d(qtt, checks)


def qtt2tensor3d(qtt: torch.Tensor, checks: bool = True) -> torch.Tensor:
    shape = qtt.shape
    if checks:
        assert all([shape[0] == shape[i] == 2 for i in range(1, len(shape))]), f'Only qtt tensors are supported'
        assert len(shape) // 3 == len(shape) / 3
    num_dims = len(shape)
    dimentions = np.arange(num_dims)
    dims = list(np.concatenate([
        dimentions[0::3],
        dimentions[1::3],
        dimentions[2::3]]))
    t = qtt.permute(dims)
    dim_grid = 2 ** (num_dims // 3)
    return t.reshape([dim_grid] * 3)


def qtt_stack(
        ts: List[tn.Tensor],
        N: int = 3,
        eps: Optional[float] = None,
        rank: Optional[int] = None,
        algorithm: str = 'svd'):
    '''
    Given a list of K tensors (shape (2^L)^N each) represented in the QTT format (shape 2^(N * L)),
    stack them along a new dimension that is interleaved with their original dimensions.

    :param ts: list of QTT's (`tntorch.Tensor`), each of shape 2^(N * L)
    :param N: number of spatial dimensions (default is 3)
    :param rmax: maximal rank of the stacked result
    :param algorithm: eig or svd (default)
    :return: a `tntorch.Tensor` of shape 2^((N + 1) * L)
    '''
    assert all([t.shape == ts[0].shape for t in ts[1:]])
    assert min(ts[0].shape) == max(ts[0].shape) == 2
    L = ts[0].dim() // N
    assert int(np.ceil(np.log2(len(ts)))) <= L

    # NOTE: Pad with zero tensors to make sure time size is 2^L
    ts = ts + [tn.zeros_like(ts[0]) for i in range(2**L - len(ts))]

    output_ts = []
    for i in range(len(ts)):
        t = tn.unsqueeze(ts[i], dim=range(N, L * (N + 1) + 1, N + 1))
        t = t.repeat(*([1] * N + [2]) * L)
        # NOTE: We will put least-signficant bits towards the left
        for l in range(L):
            t.cores[(l + 1) * (N + 1) - 1][:, int((i & (1 << l)) == 0), :] = 0
        output_ts.append(t)
    return reduce_tt(output_ts, operator.add, eps=eps, rank=rank, algorithm=algorithm)


def get_qtt_frame(qtt_scene: tn.Tensor, frame: int, N: int = 3):
    '''
    Extract a single qtt frame from a compressed qtt scene
    '''
    idxs = []
    frame_bits = np.binary_repr(frame, qtt_scene.dim() // (N + 1))[::-1]
    bit_index = 0
    for i in range(qtt_scene.dim()):
        if (i + 1) % (N + 1) == 0:
            idxs.append(int(frame_bits[bit_index]))
            bit_index += 1
        else:
            idxs.append(slice(None))

    return qtt_scene[idxs]

def get_oqtt_frame(oqtt_scene: tn.Tensor, frame: int, N: int = 3):
    '''
    Extract a single oqtt frame from a compressed oqtt scene
    '''
    idxs = []
    frame_bits = np.binary_repr(frame, qtt_scene.dim() // (N + 1))[::-1]
    bit_index = 0
    for i in range(qtt_scene.dim()):
        if (i + 1) % (N + 1) == 0:
            idxs.append(int(frame_bits[bit_index]))
            bit_index += 1
        else:
            idxs.append(slice(None))

    return qtt_scene[idxs]
