
from lib.pixielib.utils.config import cfg as pixie_cfg
from lib.pixielib.pixie import PIXIE
from lib.pixielib.models.SMPLX import SMPLX as PIXIE_SMPLX
import numpy as np
import os
import torch
optimed_betas=torch.zeros(1,10).to("cuda:0")
pixie_cfg.merge_from_list(["model.n_shape", 10, "model.n_exp", 10])
smpl_model = PIXIE_SMPLX(pixie_cfg.model).to("cuda:0")
params_fname = os.path.join("/media/star/Extreme SSD/HBW/HBW/metrics", f'spin_smplx.npz')
new_method_result = np.load(os.path.join("/media/star/Extreme SSD/HBW/HBW/metrics", f'spin.npz'))
out_params = dict()
labels = new_method_result['image_name']
shape = new_method_result['shape']

smpl_verts, smpl_landmarks, smpl_joints = smpl_model(
                            shape_params=torch.from_numpy(np.array(shape)).to("cuda:0"),
                        )
out_params["image_name"]=labels
out_params["v_shaped"]=smpl_verts.detach().cpu().numpy()
out_params["shape"]=shape
np.savez_compressed(params_fname, **out_params)