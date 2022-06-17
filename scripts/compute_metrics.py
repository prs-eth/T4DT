import os
import os.path as osp
import sys
import torch
import tqdm
import trimesh
import configargparse
import skimage
import time
import kaolin as kal

sys.path.append(osp.join(os.path.abspath(os.getcwd()), '..',))
from t4dt.metrics import hausdorff, MSDM2

MIN_TSDF = -1.
MAX_TSDF = 1.
NUM_SAMPLE_POINTS = 30000

# NOTE: we use chamfer distance, L2, IoU
parser = configargparse.ArgumentParser(description='Compress the scene of SDFs with 4D TT decomposition')
parser.add('-c', '--config', is_config_file=True, help='config file path')

parser.add('--data_dir', required=True, type=str, help='Dirrectory with data')
parser.add('--experiment_name', required=True, type=str, help='Experiment name')
parser.add('--output_dir', required=True, type=str, help='Dirrectory for output')

parser.add('--model', required=True, type=str, help='Name of the model')
parser.add('--scene', required=True, type=str, help='Name of the scene')
parser.add('--compressed_scene_path', required=True, type=str, help='Path to compressed ')

args = parser.parse_args()

frames = []
for frame in sorted(os.listdir(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed'))):
    if frame.startswith('sdf'):
        frames.append(frame)

if len(frames) == 0:
    raise ValueError('SDF must be precomputed first')

scene = torch.load(args.compressed_scene_path)
res = torch.tensor(scene.shape[:-1])

assert len(frames) == scene.shape[-1], f'Number of frames must match, got {len(frames)} and {scene.shape[-1]}'

for i in tqdm.tqdm(range(len(frames))):
    frame_pred = scene[..., i].torch()
    frame_pred.clamp_min_(MIN_TSDF)
    frame_pred.clamp_max_(MAX_TSDF)

    # NOTE: cut sdf out and .pt add .obj
    mesh_gt = trimesh.load(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed', frames[i][4:-2] + 'obj'))
    sdf_w_coords = torch.load(osp.join(args.data_dir, 'meshes', args.model, args.scene, 'posed', frames[i]))
    sdf = sdf_w_coords['sdf']
    coords = torch.tensor(sdf_w_coords['coords'])

    tqdm.tqdm.write('Marching cube started')
    t0 = time.time()
    verts, faces, _, _ = skimage.measure.marching_cubes(
        frame_pred.numpy(),
        level=0.,
        spacing=(coords[3:] - coords[:3]) / res)
    verts = torch.tensor(verts.copy())
    faces = torch.tensor(faces.copy())
    tqdm.tqdm.write(f'Marching cube finished. Took: {time.time() - t0} s.')

    mesh_pred = trimesh.base.Trimesh(vertices=verts, faces=faces)

    tqdm.tqdm.write('Sampling points started')
    t0 = time.time()
    points_gt, _ = trimesh.sample.sample_surface(mesh_gt, NUM_SAMPLE_POINTS)
    points_pred, _ = trimesh.sample.sample_surface(mesh_pred, NUM_SAMPLE_POINTS)
    tqdm.tqdm.write(f'Sampling points finished. Took: {time.time() - t0} s.')

    points_gt = torch.tensor(points_gt[None]).cuda()
    points_pred = torch.tensor(points_pred[None]).cuda()

    chamfer_distance_error = kal.metrics.pointcloud.chamfer_distance(points_gt, points_pred)[0]
    del points_gt
    del points_pred

    l2_error = torch.norm(frame_pred - sdf.clamp_min(MIN_TSDF).clamp_max(MAX_TSDF))

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

    tqdm.tqdm.write('MSDM2 computation started')
    t0 = time.time()
    MSDM2_err = MSDM2(torch.tensor(mesh_gt.vertices), torch.tensor(mesh_gt.faces), verts, faces)
    tqdm.tqdm.write(f'MSDM2 computation finished. Took: {time.time() - t0} s.')

    print(f'l2: {l2_error}, chamfer_distance: {chamfer_distance_error}, IoU: {IoU}, MSDM2: {MSDM2_err}')

    break
