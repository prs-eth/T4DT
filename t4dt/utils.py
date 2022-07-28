import torch
import torch.nn.functional as F
import skimage
import trimesh
import numba
import numpy as np
from pysdf import SDF
from typing import Callable, List

def sdf2mesh(sdf: torch.Tensor, coords: torch.Tensor) -> trimesh.base.Trimesh:
    '''
    Convert sdf to mesh using marching cubes algorithm.

    :param sdf: torch.Tensor with signed distance field
    :param coords: real world coordinates of bounding box for sdf
    :return: trimesh.base.Trimesh
    '''
    vertices, faces, _, _ = skimage.measure.marching_cubes(
    sdf.detach().cpu().numpy(),
    level=0.,
    spacing=(coords[3:] - coords[:3]) / torch.tensor(sdf.shape))

    vertices = torch.tensor(vertices.copy())
    faces = torch.tensor(faces.copy())

    # NOTE: process=False required for the correct work of MSDM2 
    return trimesh.base.Trimesh(vertices=vertices, faces=faces, process=False)


def get_3dtensors_patches(t: torch.Tensor, patch_size: int, coords: torch.Tensor) -> torch.Tensor:
    '''
    Cuts patches from tensors at certain locations.

    :param t: 3D tensor, N \times M \times L
    :param patch_size: receptieve field required, int
    :param coords: centers of patches, torch.Tensor e.g., torch.tensor([[x_1, y_1, z_1], ..., [x_b, y_b, z_b]])
    :return: torch.Tensor of size b \times patch_size \times patch_size \times patch_size
    '''
    res = [None] * coords.shape[0]

    assert patch_size % 2 != 0
    w = (patch_size - 1) // 2

    for i in range(len(res)):
        patch = t[
            max(coords[i][0] - w, 0): coords[i][0] + w + 1,
            max(coords[i][1] - w, 0): coords[i][1] + w + 1,
            max(coords[i][2] - w, 0): coords[i][2] + w + 1]
        res[i] = F.pad(
            patch,
            # NOTE: left, right, top, bottom,
            [- min(coords[i][2] - w, 0),
             - min(t.shape[2] - coords[i][2] - w - 1, 0),
             - min(coords[i][1] - w, 0),
             - min(t.shape[1] - coords[i][1] - w - 1, 0),
             - min(coords[i][0] - w, 0),
             - min(t.shape[0] - coords[i][0] - w - 1, 0)],
            mode='constant', value=0)[None, ...]
    return torch.cat(res)


def find_surface_adjacent_voxels(sdf: torch.Tensor, batch_size: int = 32368) -> torch.Tensor:
    w, h, d = sdf.shape
    result = []

    xx, yy, zz = torch.meshgrid(torch.arange(w), torch.arange(h), torch.arange(d), indexing='ij')
    points = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)

    offsets = torch.tensor([
    #    [-1, -1, -1],
    #    [-1, -1,  0],
    #    [-1, -1,  1],
    #    [-1,  0, -1],
       [-1,  0,  0],
    #    [-1,  0,  1],
    #    [-1,  1, -1],
    #    [-1,  1,  0],
    #    [-1,  1,  1],
    #    [ 0, -1, -1],
       [ 0, -1,  0],
    #    [ 0, -1,  1],
       [ 0,  0, -1],
       [ 0,  0,  0],
       [ 0,  0,  1],
    #    [ 0,  1, -1],
       [ 0,  1,  0],
    #    [ 0,  1,  1],
    #    [ 1, -1, -1],
    #    [ 1, -1,  0],
    #    [ 1, -1,  1],
    #    [ 1,  0, -1],
       [ 1,  0,  0],
    #    [ 1,  0,  1],
    #    [ 1,  1, -1],
    #    [ 1,  1,  0],
    #    [ 1,  1,  1]
       ])

    for batch_start in range(0, points.shape[0], batch_size):
        pp = points[batch_start : batch_start + batch_size, :, None]
        candidates = pp + offsets.T
        candidates.clamp_min_(0)
        candidates[:, 0, :].clamp_max_(w - 1)
        candidates[:, 1, :].clamp_max_(h - 1)
        candidates[:, 2, :].clamp_max_(d - 1)

        candidates = candidates.permute(0, 2, 1).reshape(-1, 3)
        if len(candidates) == 0:
            continue
        signs = torch.sign(sdf[candidates[:, 0], candidates[:, 1], candidates[:, 2]])
        signs = signs.reshape(-1, len(offsets))
        idxs = (signs.sum(dim=-1).abs() != signs.shape[1])

        if idxs.any():
            result.append(pp[idxs][..., 0])
    return torch.cat(result) if result else torch.zeros((0, 3))


def build_sdf_scene_reader(scene_path: str, scene_prefix: str, coords: torch.Tensor) -> Callable:
    def f(coords: torch.Tensor) -> torch.Tensor:
        '''
        SDF samples at coords

        :param coords: coordinates tensor with [frame_id, x, y, z] entries
        :return: torch.Tensor
        '''
        frames = coords[:, 0].long()
        assert torch.allclose(frames, coords[:, 0]), 'frame IDs must be integer'
        result = torch.zeros(frames.shape[0])
        for frame in frames.unique():
            input_name = osp.join(scene_path, scene_prefix + f'{frame:03}.pt')
            mesh = trimesh.load(input_name)
            sdf = SDF(mesh.vertices, mesh.faces)
            result[frames == frame] = sdf(coords[frames == frame][:, 1:])
        
        return torch.cat(result)

    return f
