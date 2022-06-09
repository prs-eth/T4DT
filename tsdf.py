import torch
from typing import List, Optional

class TSDF:
    def __init__(self, res: List[int], world: torch.Tensor, camera: torch.Tensor):
        self.res = res
        self.volume = torch.zeros(res)
        self.world = world
        self.camera = camera

    def integrate(self, rgb: torch.Tensor, depth: torch.Tensor, pose: torch.Tensor, ):
        '''
        Integrate image to TSDF

        Args:
            rgb: WxHx3 uint8 image tensor
            depth: WxH depth image
        '''
        # Fold RGB color image into a single channel image
        color_im = torch.floor(rgb[..., 2] * 256 * 256 + rgb[..., 1] * 256 + rgb[..., 0])

        # Convert world coordinates to camera coordinates
        world2cam = torch.inverse(pose)
        cam_c = (world2cam @ self.world.transpose(1, 0)).transpose(1, 0).float()

        # Convert camera coordinates to pixel coordinates
        fx, fy = self.cam_intr[0, 0], self.cam_intr[1, 1]
        cx, cy = self.cam_intr[0, 2], self.cam_intr[1, 2]
        pix_z = self.cam_c[:, 2]
        pix_x = torch.round((self.cam_c[:, 0] * fx / self.cam_c[:, 2]) + cx).long()
        pix_y = torch.round((self.cam_c[:, 1] * fy / self.cam_c[:, 2]) + cy).long()

        # Eliminate pixels outside view frustum
        valid_pix = (pix_x >= 0) & (pix_x < self.im_w) & (pix_y >= 0) & (pix_y < self.im_h) & (pix_z > 0)
        valid_vox_x = vox_coords[valid_pix, 0]
        valid_vox_y = vox_coords[valid_pix, 1]
        valid_vox_z = vox_coords[valid_pix, 2]
        valid_pix_y = pix_y[valid_pix]
        valid_pix_x = pix_x[valid_pix]
        depth_val = depth_im[pix_y[valid_pix], pix_x[valid_pix]]
