
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import util
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    This class implements methods for rasterizing a batch of heterogenous Meshes.
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """
    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2].clone()
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(
            attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1]
        )
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0    # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals

class SRenderY(nn.Module):
    def __init__(self, image_size, faces, uv_size=256):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size


        self.rasterizer = Pytorch3dRasterizer(image_size)
        # self.uv_rasterizer = Pytorch3dRasterizer(uv_size)
        # verts, faces, aux = load_obj(obj_filename)
        # uvcoords = aux.verts_uvs[None, ...]    # (N, V, 2)
        # uvfaces = faces.textures_idx[None, ...]    # (N, F, 3)
        # faces = faces.verts_idx[None, ...]  # (N, F, 3)
        self.faces = faces



    def render_depth( self,transformed_vertices):
        '''
        -- rendering depth
        '''
        transformed_vertices = transformed_vertices.clone()
        batch_size = transformed_vertices.shape[0]
        min_z=transformed_vertices[:, :, 2].min()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - min_z.clone()
        z = -transformed_vertices[:, :, 2:].repeat(1, 1, 3)
        z = z - z.min()
        z = z / (z.max()-z.min())
        # Attributes
        attributes = util.face_vertices(z, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(
            transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes
        )

        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        depth_images = rendering[:, :1, :, :].detach()
        return depth_images