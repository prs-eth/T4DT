import os
import os.path as osp
import trimesh
from pysdf import SDF
import torch
import tqdm


DATASET_DIR = '/scratch2/data/ana/'
RES = 512

scenes = {}
for scene in tqdm.tqdm(os.listdir(DATASET_DIR)):
    coords_fname = osp.join(DATASET_DIR, scene, 'meshes', 'coords.pt')
    if osp.exists(coords_fname):
        scenes[scene] = torch.load(coords_fname)['coords']
    else:
        scenes[scene] = [float('inf'), float('inf'), float('inf'), -float('inf'), -float('inf'), -float('inf')]
        for frame in tqdm.tqdm(os.listdir(osp.join(DATASET_DIR, scene, 'meshes'))):
            if frame.startswith('watertight_'):
                input_name = osp.join(DATASET_DIR, scene, 'meshes', frame)
                mesh = trimesh.load(input_name)
                x, y, z = [torch.tensor(mesh.vertices)[:, i] for i in range(3)]
                scenes[scene] = [
                    min(scenes[scene][0], min(x)), min(scenes[scene][1], min(y)), min(scenes[scene][2], min(z)),
                    max(scenes[scene][3], max(x)), max(scenes[scene][4], max(y)), max(scenes[scene][5], max(z))]
        torch.save({'coords': scenes[scene]}, coords_fname)

    xx, yy, zz = torch.meshgrid(
        torch.linspace(scenes[scene][0], scenes[scene][3], RES),
        torch.linspace(scenes[scene][1], scenes[scene][4], RES),
        torch.linspace(scenes[scene][2], scenes[scene][5], RES), indexing='ij')
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)
    for frame in tqdm.tqdm(os.listdir(osp.join(DATASET_DIR, scene, 'meshes'))):
        if not frame.startswith('sdf_watertight_') and frame.startswith('watertight_'):
            input_name = osp.join(DATASET_DIR, scene, 'meshes', frame)
            output_name = osp.join(DATASET_DIR, scene, 'meshes', 'sdf_' + frame[:-4] + '.pt')

            if osp.exists(output_name):
                continue

            mesh = trimesh.load(input_name)
            f = SDF(mesh.vertices, mesh.faces)
            sdf = torch.tensor(f(coords).reshape(RES, RES, RES))
            torch.save({'sdf': sdf, 'coords': scenes[scene]}, output_name)
