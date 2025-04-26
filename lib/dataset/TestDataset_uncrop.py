# -*- coding: utf-8 -*-
import pickle
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

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)
from virtualmarker.dataset.demo_dataset import DemoDataset
from lib.pixielib.utils.config import cfg as pixie_cfg
from lib.pixielib.pixie import PIXIE
from lib.pixielib.models.SMPLX import SMPLX as PIXIE_SMPLX
from lib.common.imutils2 import process_image
from lib.common.train_util import Format
from lib.net.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from PIL import Image
from lib.pymafx.core import path_config
from lib.pymafx.models import pymaf_net
import os
import shutil
from lib.common.config import cfg
from lib.common.render import Render
from lib.dataset.body_model import TetraSMPLModel
from lib.dataset.mesh_util import get_visibility, SMPLX
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import detection
import virtualmarker.models as models
import os.path as osp
import torch
import glob
import numpy as np
from termcolor import colored
from PIL import ImageFile
from virtualmarker.core.config import cfg as vmcfg
from virtualmarker.core.config import  update_config, init_experiment_dir
from virtualmarker.utils.funcs_utils import load_checkpoint
from virtualmarker.utils.smpl_utils import get_smpl_faces
from smpl2smplx.transfer.__main__ import smpl2smplx
import scipy.sparse as ssp
import cv2
from virtualpose.core.config import config as det_cfg
from virtualpose.core.config import update_config as det_update_config
from virtualpose.utils.transforms import inverse_affine_transform_pts_cuda
from virtualpose.utils.utils import load_backbone_validate
import virtualpose.models as det_models
import virtualpose.dataset as det_dataset
from tqdm import tqdm
import trimesh
import torch.nn as nn
from tokenhmr.lib.models import load_tokenhmr
from tokenhmr.lib.models.smpl_wrapper import SMPL
from tokenhmr.lib.utils import recursive_to
from tokenhmr.lib.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from tokenhmr.lib.utils.renderer import Renderer, cam_crop_to_full

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TestDataset:
    def __init__(self, cfg, device):

        self.image_dir = cfg["image_dir"]
        self.seg_dir = cfg["seg_dir"]
        self.use_seg = cfg["use_seg"]
        self.hps_type = "pymafx"
        self.smpl_type = "smplx"
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
        self.smpl_data = SMPLX()

        if self.hps_type == "pymafx":
            self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
            self.hps.load_state_dict(torch.load(path_config.CHECKPOINT_FILE)["model"], strict=True)
            self.hps.eval()
            pixie_cfg.merge_from_list(["model.n_shape", 10, "model.n_exp", 10])
        elif self.hps_type == "pixie":
            self.hps = PIXIE(config=pixie_cfg, device=self.device)
        elif self.hps_type == "vm":
            init_experiment_dir()
            self.smpl_faces = get_smpl_faces()
            update_config('/media/star/Extreme SSD/code/VirtualMarker-master/configs/simple3dmesh_infer/baseline.yml')
            demo_dataset=DemoDataset()
            self.hps=models.simple3dmesh.get_model(demo_dataset.vertex_num , demo_dataset.flip_pairs, vm_A=demo_dataset.vm_A,  selected_indices=demo_dataset.selected_indices)
            load_path = vmcfg.test.weight_path
            if load_path != '':
                print('==> Loading checkpoint')
                checkpoint = load_checkpoint(load_path, master=True)
                if 'model_state_dict' in checkpoint.keys():
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                if vmcfg.model.name == 'simple3dmesh' and vmcfg.model.simple3dmesh.noise_reduce:
                    self.hps.simple3dmesh.load_state_dict(state_dict, strict=True)
                else:
                    self.hps.load_state_dict(state_dict, strict=True)
                print(colored(f'Successfully load checkpoint from {load_path}.', 'green'))
            self.hps = self.hps.cuda()
            self.hps = nn.DataParallel(self.hps)
            pixie_cfg.merge_from_list(["model.n_shape", 10, "model.n_exp", 10])
        elif self.hps_type == "tokenhmr":
            self.hps, model_cfg = load_tokenhmr(checkpoint_path="/media/star/Extreme SSD/code/VS/tokenhmr/data/checkpoints/tokenhmr_model_latest.ckpt", \
                                             model_cfg="/media/star/Extreme SSD/code/VS/tokenhmr/data/checkpoints/model_config.yaml", \
                                             is_train_state=False, is_demo=True)
            # Setup model
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.hps = self.hps.to(device)
            self.hps.eval()
            self.smpl_model=self.hps.smpl



        self.smplx_model = PIXIE_SMPLX(pixie_cfg.model).to(self.device)

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

    def get_joint_setting(self, joint_category='human36'):
        joint_num =  vmcfg.dataset.num_joints
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')
        if self.input_joint_name == 'vm' and joint_num % 16:
            flip_pairs = tuple(list(self.human36_flip_pairs) + list(self.vm_flip_pairs_reindex))

        return joint_num, skeleton, flip_pairs

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

    def depth_to_voxel(self, data_dict):

        data_dict["depth_F"] = transforms.Resize(self.vol_res)(data_dict["depth_F"])
        data_dict["depth_B"] = transforms.Resize(self.vol_res)(data_dict["depth_B"])

        depth_mask = (~torch.isnan(data_dict['depth_F']))
        depth_FB = torch.cat([data_dict['depth_F'], data_dict['depth_B']], dim=0)
        depth_FB[:, ~depth_mask[0]] = 0.

        # Important: index_long = depth_value - 1
        index_z = (((depth_FB + 1.) * 0.5 * self.vol_res) - 1).clip(0, self.vol_res -
                                                                    1).permute(1, 2, 0)
        index_z_ceil = torch.ceil(index_z).long()
        index_z_floor = torch.floor(index_z).long()
        index_z_frac = torch.frac(index_z)

        index_mask = index_z[..., 0] == torch.tensor(self.vol_res * 0.5 - 1).long()
        voxels = F.one_hot(index_z_ceil[..., 0], self.vol_res) * index_z_frac[..., 0] + \
            F.one_hot(index_z_floor[..., 0], self.vol_res) * (1.0-index_z_frac[..., 0]) + \
            F.one_hot(index_z_ceil[..., 1], self.vol_res) * index_z_frac[..., 1]+ \
            F.one_hot(index_z_floor[..., 1], self.vol_res) * (1.0 - index_z_frac[..., 1])

        voxels[index_mask] *= 0
        voxels = torch.flip(voxels, [2]).permute(2, 0, 1).float()    #[x-2, y-0, z-1]

        return {
            "depth_voxels": voxels.flip([
                0,
            ]).unsqueeze(0).to(self.device),
        }

    def compute_voxel_verts(self, body_pose, global_orient, betas, trans, scale):

        smpl_path = osp.join(self.smpl_data.model_dir, "smpl/SMPL_NEUTRAL.pkl")
        tetra_path = osp.join(self.smpl_data.tedra_dir, "tetra_neutral_adult_smpl.npz")
        smpl_model = TetraSMPLModel(smpl_path, tetra_path, "adult")

        pose = torch.cat([global_orient[0], body_pose[0]], dim=0)
        smpl_model.set_params(rotation_matrix_to_angle_axis(rot6d_to_rotmat(pose)), beta=betas[0])

        verts = (
            np.concatenate([smpl_model.verts, smpl_model.verts_added], axis=0) * scale.item() +
            trans.detach().cpu().numpy()
        )
        faces = (
            np.loadtxt(
                osp.join(self.smpl_data.tedra_dir, "tetrahedrons_neutral_adult.txt"),
                dtype=np.int32,
            ) - 1
        )

        pad_v_num = int(8000 - verts.shape[0])
        pad_f_num = int(25100 - faces.shape[0])

        verts = (
            np.pad(verts, ((0, pad_v_num),
                           (0, 0)), mode="constant", constant_values=0.0).astype(np.float32) * 0.5
        )
        faces = np.pad(faces, ((0, pad_f_num), (0, 0)), mode="constant",
                       constant_values=0.0).astype(np.int32)

        verts[:, 2] *= -1.0

        voxel_dict = {
            "voxel_verts": torch.from_numpy(verts).to(self.device).unsqueeze(0).float(),
            "voxel_faces": torch.from_numpy(faces).to(self.device).unsqueeze(0).long(),
            "pad_v_num": torch.tensor(pad_v_num).to(self.device).unsqueeze(0).long(),
            "pad_f_num": torch.tensor(pad_f_num).to(self.device).unsqueeze(0).long(),
        }

        return voxel_dict

    def __getitem__(self, index):

        img_path = self.subject_list[index]
        img_name = img_path.split("/")[-1].rsplit(".", 1)[0]

        arr_dict = process_image(img_path, self.hps_type, self.single, 512, self.detector)
        arr_dict.update({"name": img_name})



        with torch.no_grad():
            if self.hps_type == "pixie":
                preds_dict = self.hps.forward(arr_dict["img_hps"].to(self.device))
            elif self.hps_type == 'pymafx':
                batch = {k: v.to(self.device) for k, v in arr_dict["img_pymafx"].items()}
                print(batch)
                preds_dict, _ = self.hps.forward(batch)
            elif self.hps_type == "vm":

                input_path= img_path.replace( img_path.split("/")[-2],"temp")
                path=""
                for j in range(len(img_path.split("/"))-2):
                    path=os.path.join(path, img_path.split("/")[j])
                os.makedirs(os.path.join("/",path,"temp"), exist_ok=True)
                image_save = (arr_dict['img_icon'].squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255
                mask_save = (arr_dict['img_mask'].squeeze().cpu().numpy())
                image_save[:, :, 0] = mask_save * image_save[:, :, 0]
                image_save[:, :, 1] = mask_save * image_save[:, :, 1]
                image_save[:, :, 2] = mask_save * image_save[:, :, 2]
                image_save = Image.fromarray(image_save.astype(np.uint8))
                image_save.save(input_path)

                img_dir = osp.join(osp.dirname(osp.abspath(input_path)), 'VM', input_path.split('/')[-1][:-4])
                os.makedirs(img_dir, exist_ok=True)
                shutil.copy(input_path, img_dir)
                img_path_list = [osp.join(img_dir, input_path.split('/')[-1])]
                fps = -1
                detection_all, max_person, valid_frame_idx_all = detect_all_persons( img_dir)
                demo_dataset = DemoDataset(img_path_list, detection_all)
                meta_data=demo_dataset.get_image_info(0)
                imgs = meta_data['img'].unsqueeze(0).cuda()

                inv_trans, intrinsic_param = torch.from_numpy(meta_data['inv_trans']).unsqueeze(0).cuda(), torch.from_numpy(meta_data['intrinsic_param']).unsqueeze(0).cuda()
                pose_root = torch.from_numpy(meta_data['root_cam']).unsqueeze(0).cuda()
                depth_factor = torch.from_numpy(meta_data['depth_factor']).unsqueeze(0).cuda()


                self.hps.eval()
                with torch.no_grad():
                    _, _, _, _, pred_mesh, _, pred_root_xy_img = self.hps(imgs, inv_trans, intrinsic_param, pose_root,
                                                                        depth_factor, flip_item=None, flip_mask=None)

                output=smpl2smplx(pred_mesh/1000.0,
                           torch.from_numpy( self.smpl_faces.astype(np.int32)).unsqueeze(0).to(self.device),
                           "/media/star/Extreme SSD/code/VS/vm_smplx.obj")

                # smplx_mesh=trimesh.load_mesh("/media/star/dataset_SSD/dataset/THUman.20 Release Smpl-X Paras/0015/mesh_smplx.obj")
                # output = smpl2smplx(torch.from_numpy(np.array(smplx_mesh.vertices)).to(self.device).unsqueeze(0).float(),
                #                     torch.from_numpy(self.smpl_faces.astype(np.int32)).unsqueeze(0).to(self.device),
                #                     "/media/star/Extreme SSD/code/VS/vm_smplx.obj")

                # f=open("/media/star/dataset_SSD/dataset/THUman.20 Release Smpl-X Paras/0015/smplx_param.pkl",'rb')
                # output = pickle.load(f)
                # print(data.keys())
                # for i in output.keys():
                #     print(i, output[i].shape)

                os.makedirs(osp.join(self.out_dir, cfg.name, "obj"), exist_ok=True)
                trimesh.Trimesh(pred_mesh.squeeze().cpu().numpy() / 1000.0, self.smpl_faces).export( osp.join(self.out_dir, cfg.name, "obj",img_name+"smpl.obj")
                    )
            elif self.hps_type == "tokenhmr":

                batch = {}
                batch['img']=arr_dict["img_token"].to(self.device)

                with torch.no_grad():
                    preds_dict = self.hps(batch)
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
            trimesh.Trimesh(arr_dict["smpl_verts"].squeeze().cpu().numpy() ).export(
                "/media/star/Extreme SSD/code/VS/vm_smpl.obj")

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
        elif self.hps_type == "vm":
            arr_dict["betas"] = output["betas"].to(self.device)
            arr_dict["body_pose"] = output["body_pose"].to(self.device)
            arr_dict["global_orient"] = output["global_orient"].to(self.device)
            # arr_dict["smpl_verts"] = output["vertices"].to(self.device)
            arr_dict["left_hand_pose"] = output["left_hand_pose"].to(self.device)
            arr_dict["right_hand_pose"] = output["right_hand_pose"].to(self.device)

            arr_dict['jaw_pose'] = output['jaw_pose'].to(self.device)
            # arr_dict["exp"] = output["expression"].to(self.device)
            arr_dict["exp"] =torch.zeros_like(output["expression"]).to(self.device)
            output["transl"]=output["transl"].detach()
            tranX, tranY = output["transl"][:, :2].split(1, dim=1)

            # arr_dict["betas"] = torch.from_numpy(output["betas"]).to(self.device)
            # arr_dict["body_pose"] = torch.from_numpy(output["body_pose"]).to(self.device)
            # arr_dict["global_orient"] = torch.from_numpy(output["global_orient"]).to(self.device)
            # # arr_dict["smpl_verts"] = output["vertices"].to(self.device)
            # arr_dict["left_hand_pose"] = torch.from_numpy(output["left_hand_pose"]).to(self.device)
            # arr_dict["right_hand_pose"] = torch.from_numpy(output["right_hand_pose"]).to(self.device)
            #
            # arr_dict['jaw_pose'] = torch.from_numpy(output['jaw_pose']).to(self.device)
            # arr_dict["exp"] = torch.from_numpy(output["expression"]).to(self.device)


            # tranX=torch.tensor(np.array([0])).to(self.device)
            # tranY=torch.tensor(np.array([0])).to(self.device)
            scale=torch.tensor(np.array([1])).to(self.device)

            N_body, N_pose = arr_dict["body_pose"].shape[:2]
            arr_dict["body_pose"] = arr_dict["body_pose"][:, :, :, :2].reshape(N_body, N_pose, -1)
            arr_dict["global_orient"] = arr_dict["global_orient"][:, :, :, :2].reshape(N_body, 1, -1)
        elif self.hps_type == "tokenhmr":
            arr_dict.update(preds_dict)
            arr_dict["global_orient"] = preds_dict["global_orient"]
            arr_dict["betas"] = preds_dict["betas"]    #200
            arr_dict["body_pose"] = preds_dict["body_pose"]
            arr_dict["smpl_verts"] = preds_dict["pred_vertices"]
            scale, tranX, tranY = preds_dict["pred_cam"].split(1, dim=1)
            trimesh.Trimesh(arr_dict["smpl_verts"].squeeze().cpu().numpy()).export(
                "/media/star/Extreme SSD/code/VS/vm_smpl.obj")



        arr_dict["scale"] = scale.unsqueeze(1)
        arr_dict["trans"] = (
            torch.cat([tranX, tranY, torch.zeros_like(tranX)],
                      dim=1).unsqueeze(1).to(self.device).float()
        )




        # data_dict info (key-shape):
        # scale, tranX, tranY - tensor.float
        # betas - [1,10] / [1, 200]
        # body_pose - [1, 23, 3, 3] / [1, 21, 3, 3]
        # global_orient - [1, 1, 3, 3]
        # smpl_verts - [1, 6890, 3] / [1, 10475, 3]

        # from rot_mat to rot_6d for better optimization


        return arr_dict

    def render_normal(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="rgb")



    def render_depth(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="depth")

    # def render_depth(self, verts, faces):
    #     from lib.common.render_icon import Render
    #     render = Render(size=512, device=self.device)
    #
    #     # render optimized mesh (normal, T_normal, image [-1,1])
    #     render.load_meshes(verts, faces)
    #     return render.get_depth_map(cam_ids=[0, 2])

    # def render_depth(self, verts, faces):
    #     from lib.common.render_depth import SRenderY
    #     self.render2 = SRenderY(512,faces)
    #
    #     # render optimized mesh (normal, T_normal, image [-1,1])
    #     # render.load_meshes(verts, faces)
    #     return self.render2.render_depth( verts)


def detect_all_persons( img_dir):
    # prepare detection model
    virtualpose_name = 'VirtualPose'
    det_update_config(
        f'/media/star/Extreme SSD/code/VS/{virtualpose_name}/configs/images/images_inference.yaml')

    det_model = eval('det_models.multi_person_posenet.get_multi_person_pose_net')(det_cfg, is_train=False)
    with torch.no_grad():
        det_model = torch.nn.DataParallel(det_model.cuda())

    pretrained_file = osp.join("/media/star/Extreme SSD/code/VS", f'{virtualpose_name}',
                               det_cfg.NETWORK.PRETRAINED)
    state_dict = torch.load(pretrained_file)
    new_state_dict = {k: v for k, v in state_dict.items() if 'backbone.pose_branch.' not in k}
    det_model.module.load_state_dict(new_state_dict, strict=False)
    pretrained_file = osp.join("/media/star/Extreme SSD/code/VS", f'{virtualpose_name}', det_cfg.NETWORK.PRETRAINED_BACKBONE)
    det_model = load_backbone_validate(det_model, pretrained_file)
    # prepare detection dataset
    infer_dataset = det_dataset.images(
        det_cfg, img_dir, focal_length=1700,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]))
    infer_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=det_cfg.TEST.BATCH_SIZE * 1,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    det_model.eval()

    max_person = 0
    detection_all = []
    valid_frame_idx_all = []

    with torch.no_grad():
        for _, (inputs, targets_2d, weights_2d, targets_3d, meta, input_AGR) in enumerate(tqdm(infer_loader, dynamic_ncols=True)):

            _, _, output, _, _ = det_model(views=inputs, meta=meta, targets_2d=targets_2d,
                                           weights_2d=weights_2d, targets_3d=targets_3d, input_AGR=input_AGR)
            det_results, n_person, valid_frame_idx = output2original_scale(meta, output)
            detection_all += det_results
            valid_frame_idx_all += valid_frame_idx
            max_person = max(n_person, max_person)

    # list to array
    detection_all = np.array(detection_all)  # (N*T, 8)
    return detection_all, max_person, valid_frame_idx_all

def get_joint_setting(self, joint_category='human36'):
    joint_num =  cfg.dataset.num_joints
    skeleton = eval(f'self.{joint_category}_skeleton')
    flip_pairs = eval(f'self.{joint_category}_flip_pairs')
    if self.input_joint_name == 'vm' and joint_num % 16:
        flip_pairs = tuple(list(self.human36_flip_pairs) + list(self.vm_flip_pairs_reindex))

    return joint_num, skeleton, flip_pairs


def output2original_scale(meta, output, vis=False):
    img_paths, trans_batch = meta['image'], meta['trans']
    bbox_batch, depth_batch, roots_2d = output['bboxes'], output['depths'], output['roots_2d']

    scale = torch.tensor((det_cfg.NETWORK.IMAGE_SIZE[0] / det_cfg.NETWORK.HEATMAP_SIZE[0], \
                          det_cfg.NETWORK.IMAGE_SIZE[1] / det_cfg.NETWORK.HEATMAP_SIZE[1]), \
                         device=bbox_batch.device, dtype=torch.float32)

    det_results = []
    valid_frame_idx = []
    max_person = 0
    for i, img_path in enumerate(img_paths):
        if vis:
            img = cv2.imread(img_path)

        frame_id = 0

        trans = trans_batch[i].to(bbox_batch[i].device).float()

        n_person = 0
        for bbox, depth, root_2d in zip(bbox_batch[i], depth_batch[i], roots_2d[i]):
            if torch.all(bbox == 0):
                break
            bbox = (bbox.view(-1, 2) * scale[None, [1, 0]]).view(-1)
            root_2d *= scale[[1, 0]]
            bbox_origin = inverse_affine_transform_pts_cuda(bbox.view(-1, 2), trans).reshape(-1)
            roots_2d_origin = inverse_affine_transform_pts_cuda(root_2d.view(-1, 2), trans).reshape(-1)

            # frame_id, x_min, y_min, x_max, y_max, pixel_root_x, pixel_root_y, depth
            det_results.append([
                                   frame_id] + bbox_origin.cpu().numpy().tolist() + roots_2d_origin.cpu().numpy().tolist() + depth.cpu().numpy().tolist())

            if vis:
                img = cv2.putText(img, '%.2fmm' % depth, (int(bbox_origin[0]), int(bbox_origin[1] - 5)), \
                                  cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                img = cv2.rectangle(img, (int(bbox_origin[0]), int(bbox_origin[1])),
                                    (int(bbox_origin[2]), int(bbox_origin[3])), \
                                    (255, 0, 0), 1)
                img = cv2.circle(img, (int(roots_2d_origin[0]), int(roots_2d_origin[1])), 5, (0, 0, 255), -1)
            n_person += 1

        if vis:
            cv2.imwrite(f'{cfg.vis_dir}/origin_det_{i}.jpg', img)
        max_person = max(n_person, max_person)
        if n_person:
            valid_frame_idx.append(frame_id)
    return det_results, max_person, valid_frame_idx
