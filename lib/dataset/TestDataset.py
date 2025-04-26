# -*- coding: utf-8 -*-
import pickle
import random
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import warnings
import logging
import os
warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

from lib.pixielib.models.SMPLX import SMPLX as PIXIE_SMPLX
from lib.common.imutils_recloth import process_image
from lib.common.train_util import Format
from lib.net.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from PIL import Image
import random
from lib.common.config import cfg
from lib.common.render import Render
from lib.dataset.body_model import TetraSMPLModel
from lib.dataset.mesh_util import get_visibility, SMPLX
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import detection
from spin.models import hmr
from tokenhmr.lib.models.smpl_wrapper import SMPL
import os.path as osp
import torch
import glob
import numpy as np
from termcolor import colored
from PIL import ImageFile
from tokenhmr.lib.configs import get_config
import cv2
from tqdm import tqdm
import trimesh
import torch.nn as nn


ImageFile.LOAD_TRUNCATED_IMAGES = True


class TestDataset:
    def __init__(self, cfg, device):

        self.image_dir = cfg["image_dir"]
        self.seg_dir = cfg["seg_dir"]
        self.use_seg = cfg["use_seg"]
        self.hps_type = "tokenhmr"
        self.smpl_type = "smpl"
        self.smpl_gender = "neutral"
        self.vol_res = cfg["vol_res"]
        self.single = cfg["single"]
        self.out_dir = cfg["out_dir"]

        self.device = device

        keep_lst = sorted(glob.glob(f"{self.image_dir}/*"))
        img_fmts = ["jpg", "png", "jpeg", "JPG", "bmp", "exr"]

        self.subject_list = sorted(
            [item for item in keep_lst if item.split(".")[-1] in img_fmts], reverse=False
        )


        # smpl related
        

        if self.hps_type == "pymafx":
            from lib.pixielib.utils.config import cfg as pixie_cfg
            from lib.pixielib.pixie import PIXIE
            from lib.pymafx.core import path_config
            from lib.pymafx.models import pymaf_net
            self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
            self.hps.load_state_dict(torch.load(path_config.CHECKPOINT_FILE)["model"], strict=True)
            self.hps.eval()
            pixie_cfg.merge_from_list(["model.n_shape", 10, "model.n_exp", 10])
            self.smplx_model = PIXIE_SMPLX(pixie_cfg.model).to(self.device)
            
        # elif self.hps_type == "pixie":
        #     self.hps = PIXIE(config=pixie_cfg, device=self.device)
        #     self.smplx_model = PIXIE_SMPLX(pixie_cfg.model).to(self.device)
            
        elif self.hps_type == "tokenhmr":
            from tokenhmr.lib.models import load_tokenhmr
            
            self.hps, model_cfg = load_tokenhmr(checkpoint_path="/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/data/tokenhmr/checkpoints/tokenhmr_model_latest.ckpt", \
                                                model_cfg="/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/data/tokenhmr/checkpoints/model_config.yaml", \
                                                is_train_state=False, is_demo=True)
            # Setup model
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.hps = self.hps.to(device)
            self.hps.eval()
            self.smpl_model=self.hps.smpl
        elif self.hps_type == "spin":
            
            
            smpl_model_cfg = get_config("/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/data/tokenhmr/checkpoints/model_config.yaml")
            smpl_cfg = {k.lower(): v for k,v in dict(smpl_model_cfg.SMPL).items()}
            self.smpl_model = SMPL(**smpl_cfg).to(device=self.device)
    

            self.hps = hmr('/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/data/spin/data/smpl_mean_params.npz').to(device)
            checkpoint = torch.load("/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/data/spin/data/model_checkpoint.pt")
            self.hps.load_state_dict(checkpoint['model'], strict=False)
            
            self.hps.eval()
        elif self.hps_type == "pymaf":
            from pymaf.models import pymaf_net
            from pymaf.core import cfgs
            cfgs.update_cfg("/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/pymaf/configs/pymaf_config.yaml")
            
            self.hps = pymaf_net("/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/data/pymaf/smpl_mean_params.npz", pretrained=True).to(device)
            checkpoint = torch.load("/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/data/pymaf/PyMAF_model_checkpoint.pt")

            self.hps.load_state_dict(checkpoint['model'], strict=True)
            self.hps.eval()

            smpl_model_cfg = get_config("/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/data/tokenhmr/checkpoints/model_config.yaml")
            smpl_cfg = {k.lower(): v for k,v in dict(smpl_model_cfg.SMPL).items()}
            self.smpl_model = SMPL(**smpl_cfg).to(device=self.device)
    
        
        
        
        self.smpl_data = SMPLX()



        

        self.detector = detection.maskrcnn_resnet50_fpn(
            weights=detection.MaskRCNN_ResNet50_FPN_V2_Weights
        )
        self.detector.eval()

        print(
            colored(
                f"SMPL-X estimate with {Format.start} {self.hps_type.upper()} {Format.end}", "green"
            )
        )

        self.render = Render(size=512, device=self.device)

    def __len__(self):
        return len(self.subject_list)



    def compute_vis_cmap(self, smpl_verts, smpl_faces):

        (xy, z) = torch.as_tensor(smpl_verts).split([2, 1], dim=-1)
        smpl_vis = get_visibility(xy, z,
                                  torch.as_tensor(smpl_faces).long()[:, :,
                                                                     [0, 2, 1]]).unsqueeze(-1)
        smpl_cmap = self.smpl_data.cmap_smpl_vids(self.smpl_type).unsqueeze(0)

        return {
            "smpl_vis": smpl_vis.to(self.device),
            "smpl_cmap": smpl_cmap.to(self.device),
            "smpl_verts": smpl_verts,
        }




    def __getitem__(self, index):

        img_path = self.subject_list[index]
        img_name = img_path.split("/")[-1].rsplit(".", 1)[0]

        arr_dict = process_image(img_path, self.hps_type, self.single, 512, self.detector)
        arr_dict.update({"name": img_name})
        arr_dict.update({"img_path": img_path})



        with torch.no_grad():
            if self.hps_type == "pixie":
                preds_dict = self.hps.forward(arr_dict["img_hps"].to(self.device))
            elif self.hps_type == 'pymafx':
                batch = {k: v.to(self.device) for k, v in arr_dict["img_pymafx"].items()}
                preds_dict, _ = self.hps.forward(batch)
            elif self.hps_type == "tokenhmr":

                batch = {}
                batch['img']=arr_dict["img_token"].to(self.device)

                with torch.no_grad():
                    preds_dict = self.hps(batch)
            elif self.hps_type == "spin":
                pred_rotmat, pred_betas, pred_camera = self.hps(arr_dict["img_hps"].to(self.device))
            elif self.hps_type=='pymaf':
                preds_dict, _ = self.hps(arr_dict["img_hps"].to(self.device))
                


        if self.smpl_type=="smplx":

            arr_dict["smpl_faces"] = (
                torch.as_tensor(self.smpl_data.smplx_faces.astype(np.int64)).unsqueeze(0).long().to(
                    self.device
                )
            )
        elif self.smpl_type=="smpl":
            arr_dict["smpl_faces"] = (
                torch.as_tensor(self.smpl_data.smpl_faces.astype(np.int64)).unsqueeze(0).long().to(
                    self.device
                )
            )
        arr_dict["type"] = self.smpl_type

        if self.hps_type == "pymafx":
            output = preds_dict["mesh_out"][-1]
            scale, tranX, tranY = output["theta"][:, :3].split(1, dim=1)
            arr_dict["betas"] = output["pred_shape"]
            arr_dict["body_pose"] = output["rotmat"][:, 1:22]
            arr_dict["global_orient"] = output["rotmat"][:, 0:1]
            arr_dict["smpl_verts"] = output["smplx_verts"]
            arr_dict["left_hand_pose"] = output["pred_lhand_rotmat"]
            arr_dict["right_hand_pose"] = output["pred_rhand_rotmat"]
            arr_dict['jaw_pose'] = output['pred_face_rotmat'][:, 0:1]
            arr_dict["exp"] = output["pred_exp"]
            # 1.2009, 0.0013, 0.3954

            N_body, N_pose = arr_dict["body_pose"].shape[:2]
            arr_dict["body_pose"] = arr_dict["body_pose"][:, :, :, :2].reshape(N_body, N_pose, -1)
            arr_dict["global_orient"] = arr_dict["global_orient"][:, :, :, :2].reshape(N_body, 1, -1)

            # print(tranX, tranY)




        elif self.hps_type == "pixie":
            arr_dict.update(preds_dict)
            arr_dict["global_orient"] = preds_dict["global_pose"]
            arr_dict["betas"] = preds_dict["shape"]    #200
            arr_dict["smpl_verts"] = preds_dict["vertices"]
            scale, tranX, tranY = preds_dict["cam"].split(1, dim=1)

            N_body, N_pose = arr_dict["body_pose"].shape[:2]
            arr_dict["body_pose"] = arr_dict["body_pose"][:, :, :, :2].reshape(N_body, N_pose, -1)
            arr_dict["global_orient"] = arr_dict["global_orient"][:, :, :, :2].reshape(N_body, 1, -1)

            # 1.1435, 0.0128, 0.3520

        elif self.hps_type == "tokenhmr":
            arr_dict.update(preds_dict)
            arr_dict["global_orient"] = preds_dict["global_orient"]
            arr_dict["betas"] = preds_dict["betas"]    #200
            arr_dict["body_pose"] = preds_dict["body_pose"]
            arr_dict["smpl_verts"] = preds_dict["pred_vertices"]
            _, tranX, tranY = preds_dict["pred_cam"].split(1, dim=1)
            scale = torch.tensor(np.array([1])).to(self.device).float()
            arr_dict["scale"] = scale.unsqueeze(1).float()
            arr_dict["trans"] = (
                torch.cat([tranX, tranY, torch.zeros_like(tranX)],
                        dim=1).unsqueeze(1).to(self.device).float()
            )
        
        elif self.hps_type=="spin":
            # pred_rotmat, pred_betas, pred_camera
            arr_dict["global_orient"] = pred_rotmat[:,0].unsqueeze(1)
            arr_dict["betas"] = pred_betas   #200
            arr_dict["body_pose"] = pred_rotmat[:,1:]

            
            _, tranX, tranY = pred_camera.split(1, dim=1)
            scale = torch.tensor(np.array([1])).to(self.device).float()
            arr_dict["scale"] = scale.unsqueeze(1).float()
            arr_dict["trans"] = (
                torch.cat([tranX, tranY, torch.zeros_like(tranX)],
                        dim=1).unsqueeze(1).to(self.device).float()
            )

        elif self.hps_type=="pymaf":
            # pred_rotmat = preds_dict['smpl_out'][-1]['rotmat'].contiguous().view(-1, 24, 3, 3)
            # pred_betas = preds_dict['smpl_out'][-1]['theta'][:, 3:13].contiguous()
            # pred_camera = preds_dict['smpl_out'][-1]['theta'][:, :3].contiguous()

            output = preds_dict['smpl_out'][-1]
            scale, tranX, tranY = output['theta'][0, :3]
            arr_dict['betas'] = output['pred_shape']
            arr_dict['body_pose'] = output['rotmat'][:, 1:]
            arr_dict['global_orient'] = output['rotmat'][:, 0:1]
            arr_dict['smpl_verts'] = output['verts']
            # arr_dict["scale"] = scale.unsqueeze(1).float()
            # arr_dict["trans"] = (
            #     torch.cat([tranX, tranY, torch.zeros_like(tranX)],
            #             dim=1).unsqueeze(1).to(self.device).float()
            # )
            arr_dict['scale'] = scale
            arr_dict['trans'] = torch.tensor([tranX, tranY,0.0]).unsqueeze(0).to(self.device).float()


       





        return arr_dict

    def render_normal(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="rgb")



    def render_depth(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="depth")









