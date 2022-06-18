import torch
import sys
import os
import os.path as osp
import numpy as np
import igl

sys.path.append(osp.join(osp.abspath(os.getcwd()), '..', 'src', 'build'))

try:
    import py_mepp2
except:
    raise RuntimeError('Compile py_mepp2 first')


def hausdorff(verts_a: torch.Tensor, faces_a: torch.Tensor, verts_b: torch.Tensor, faces_b: torch.Tensor):
    return igl.hausdorff(
        verts_a.detach().cpu().numpy().astype(np.float64),
        faces_a.detach().cpu().numpy().astype(np.int64),
        verts_b.detach().cpu().numpy().astype(np.float64),
        faces_b.detach().cpu().numpy().astype(np.int64))


def MSDM2(verts_a: torch.Tensor, faces_a: torch.Tensor, verts_b: torch.Tensor, faces_b: torch.Tensor):
    return py_mepp2.MSDM2(
        verts_a.detach().cpu().numpy().astype(np.float64),
        faces_a.detach().cpu().numpy().astype(np.int64),
        verts_b.detach().cpu().numpy().astype(np.float64),
        faces_b.detach().cpu().numpy().astype(np.int64))
