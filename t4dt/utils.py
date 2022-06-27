import torch
import skimage
import trimesh

def sdf2mesh(sdf: torch.Tensor, coords: torch.Tensor, res: int):
    vertices, faces, _, _ = skimage.measure.marching_cubes(
    tsdf.detach().cpu().numpy(),
    level=0.,
    spacing=(coords[3:] - coords[:3]) / res)

    vertices = torch.tensor(vertices.copy())
    faces = torch.tensor(faces.copy())
    return trimesh.base.Trimesh(vertices=vertices, faces=faces)