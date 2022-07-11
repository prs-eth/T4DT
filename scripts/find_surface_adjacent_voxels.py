import os
import os.path as osp
import trimesh
import torch
import tqdm
import sys

sys.path.append(osp.join(os.path.abspath(os.getcwd()), '..',))
from t4dt.utils import find_surface_adjacent_voxels


DATASET_DIR = '/scratch2/data/cape_release/'

for model in tqdm.tqdm(os.listdir(osp.join(DATASET_DIR, 'meshes'))):
    for scene in tqdm.tqdm(os.listdir(osp.join(DATASET_DIR, 'meshes', model))):
        output_fname = osp.join(DATASET_DIR, 'meshes', model, scene, 'near_surface_voxel_idxs_cross.pt')
        result = {}
        if osp.exists(output_fname):
                continue
        frames = []
        for frame in sorted(os.listdir(osp.join(DATASET_DIR, 'meshes', model, scene, 'posed'))):
            if frame.startswith('sdf'):
                frames.append(frame)
        for frame in tqdm.tqdm(frames):
            if frame.startswith('sdf_watertight_'):
                input_name = osp.join(DATASET_DIR, 'meshes', model, scene, 'posed', frame)
                sdf = torch.load(input_name)['sdf']
                result[frame] = find_surface_adjacent_voxels(sdf)
        
        torch.save(result, output_fname)
