import torch
import sys
import os
import os.path as osp
import numpy as np
import igl
import skimage
import time
import trimesh
import tqdm
import kaolin as kal
import tntorch as tn
from typing import List

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


def compute_metrics(
        frames: List[str],
        compressed_scene: tn.Tensor,
        min_tsdf: float,
        max_tsdf: float,
        num_sample_points: int,
        sample_frames: List[int]):
    assert compressed_scene.shape[-1] == len(frames)

    res = torch.tensor(compressed_scene.shape[:-1])
    result = {}
    for i in tqdm.tqdm(sample_frames):
        result[i] = {}
        frame_pred = compressed_scene[..., i].torch()
        frame_pred.clamp_min_(min_tsdf)
        frame_pred.clamp_max_(max_tsdf)

        sdf_w_coords = torch.load(frames[i][0])
        sdf = sdf_w_coords['sdf']
        coords = torch.tensor(sdf_w_coords['coords'])

        tqdm.tqdm.write('Marching cube started')
        t0 = time.time()
        verts, faces, _, _ = skimage.measure.marching_cubes(
            frame_pred.detach().numpy(),
            level=0.,
            spacing=(coords[3:] - coords[:3]) / res)
        verts = torch.tensor(verts.copy())
        faces = torch.tensor(faces.copy())
        tqdm.tqdm.write(f'Marching cube finished. Took: {time.time() - t0} s.')
        mesh_pred = trimesh.base.Trimesh(vertices=verts, faces=faces)

        tqdm.tqdm.write('Marching cube started')
        t0 = time.time()
        verts1, faces1, _, _ = skimage.measure.marching_cubes(
            sdf.detach().numpy(),
            level=0.,
            spacing=(coords[3:] - coords[:3]) / res)
        verts1 = torch.tensor(verts1.copy())
        faces1 = torch.tensor(faces1.copy())
        tqdm.tqdm.write(f'Marching cube finished. Took: {time.time() - t0} s.')

        for j, mesh_gt in enumerate([trimesh.base.Trimesh(vertices=verts1, faces=faces1), trimesh.load(frames[i][1])]):
            result[i][j] = {}
            tqdm.tqdm.write('Sampling points started')
            t0 = time.time()
            points_gt, _ = trimesh.sample.sample_surface(mesh_gt, num_sample_points)
            points_pred, _ = trimesh.sample.sample_surface(mesh_pred, num_sample_points)
            tqdm.tqdm.write(f'Sampling points finished. Took: {time.time() - t0} s.')

            points_gt = torch.tensor(points_gt[None]).cuda()
            points_pred = torch.tensor(points_pred[None]).cuda()

            chamfer_distance_error = kal.metrics.pointcloud.chamfer_distance(points_gt, points_pred)[0].detach().cpu()
            del points_gt
            del points_pred

            l2_error = torch.norm(frame_pred - sdf.clamp_min(min_tsdf).clamp_max(max_tsdf))

            tqdm.tqdm.write('Voxelgrid conversion started')
            t0 = time.time()
            vg_pred = kal.ops.conversions.trianglemeshes_to_voxelgrids(verts[None], faces, res.max().item())
            vg_gt = kal.ops.conversions.trianglemeshes_to_voxelgrids(
                torch.tensor(mesh_gt.vertices[None]),
                torch.tensor(mesh_gt.faces),
                res.max().item())
            tqdm.tqdm.write(f'Voxelgrid conversion finished. Took: {time.time() - t0} s.')

            IoU = kal.metrics.voxelgrid.iou(vg_pred, vg_gt)
            del vg_pred
            del vg_gt

            tqdm.tqdm.write('hausdorff computation started')
            t0 = time.time()
            hausdorff_dist = hausdorff(torch.tensor(mesh_gt.vertices), torch.tensor(mesh_gt.faces), verts, faces)
            tqdm.tqdm.write(f'hausdorff computation finished. Took: {time.time() - t0} s.')

            tqdm.tqdm.write('MSDM2 computation started')
            t0 = time.time()
            MSDM2_err = MSDM2(torch.tensor(mesh_gt.vertices), torch.tensor(mesh_gt.faces), verts, faces)
            tqdm.tqdm.write(f'MSDM2 computation finished. Took: {time.time() - t0} s.')
            result[i][j] = {
                'l2': l2_error,
                'chamfer_distance': chamfer_distance_error,
                'IoU': IoU[0],
                'hausdorff': hausdorff_dist,
                'MSDM2': MSDM2_err}

    return result
