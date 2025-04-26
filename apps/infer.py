# -*- coding: utf-8 -*-
import copy
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
# import sys
# sys.path.append('/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR')
import warnings
import logging
from smpl import SMPL
import cv2
import pickle as pkl
warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import torch, torchvision
import trimesh
import numpy as np
import argparse

from tqdm.auto import tqdm
from Normal import Normal
from IFGeo import IFGeo
from lib.common.config import cfg
from lib.common.train_util import init_loss, Format
from lib.dataset.TestDataset import TestDataset
from lib.dataset.mesh_util import *


torch.backends.cudnn.benchmark = True
from deal_joints import sapiens_joints_tokenhmr

if __name__ == "__main__":
            parser = argparse.ArgumentParser()

            parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
            parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=50)
            parser.add_argument("-patience", "--patience", type=int, default=5)
            parser.add_argument("-vis_freq", "--vis_freq", type=int, default=1000)
            parser.add_argument("-loop_cloth", "--loop_cloth", type=int, default=0)
            parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples/images/frame_10.png")
            parser.add_argument("-out_dir", "--out_dir", type=str, default="./examples/images/output")
            parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
            parser.add_argument("-cfg", "--config", type=str, default="/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/configs/econ.yaml")
            parser.add_argument( "--model_type", type=str, default="smpl")
            args = parser.parse_args()

            # cfg read and merge
            cfg.merge_from_file(args.config)
            cfg.merge_from_file("/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/lib/pymafx/configs/pymafx_config.yaml")

            device = torch.device(f"cuda:{args.gpu_device}")

            # setting for testing on in-the-wild images
            cfg_show_list = [
                "test_gpus", [args.gpu_device], "mcube_res", 512, "clean_mesh", True, "test_mode", True,
                "batch_size", 1
            ]

            cfg.merge_from_list(cfg_show_list)
            cfg.freeze()

            # load normal model
            normal_net = Normal.load_from_checkpoint(
                cfg=cfg, checkpoint_path="/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/data/ckpt/normal.ckpt", map_location=device, strict=False
            )
            normal_net = normal_net.to(device)
            normal_net.netG.eval()
            print(
                colored(
                    f"Resume Normal Estimator from {Format.start} {cfg.normal_path} {Format.end}", "green"
                )
            )

            # SMPLX object
            SMPLX_object = SMPLX()

            dataset_param = {
                "image_dir": args.in_dir,
                "seg_dir": args.seg_dir,
                "use_seg": True,    # w/ or w/o segmentation
                "hps_type": cfg.bni.hps_type,    # pymafx/pixie
                "vol_res": cfg.vol_res,
                "single": True,
                "out_dir": args.out_dir
            }

            if cfg.bni.use_ifnet:
                # load IFGeo model
                ifnet = IFGeo.load_from_checkpoint(
                    cfg=cfg, checkpoint_path=cfg.ifnet_path, map_location=device, strict=False
                )
                ifnet = ifnet.to(device)
                ifnet.netG.eval()

                print(colored(f"Resume IF-Net+ from {Format.start} {cfg.ifnet_path} {Format.end}", "green"))
                print(colored(f"Complete with {Format.start} IF-Nets+ (Implicit) {Format.end}", "green"))
            else:
                print(colored(f"Complete with {Format.start} SMPL-X (Explicit) {Format.end}", "green"))

            dataset = TestDataset(dataset_param, device)

            print(colored(f"Dataset Size: {len(dataset)}", "green"))

            pbar = tqdm(dataset)
           


            for data in pbar:


                losses = init_loss()

                pbar.set_description(f"{data['name']}")


                os.makedirs(osp.join(args.out_dir,"png"), exist_ok=True)



                os.makedirs(osp.join(args.out_dir, "obj"), exist_ok=True)

                in_tensor = {
                    "smpl_faces": data["smpl_faces"],
                    "image": data["img_icon"].to(device),
                    "mask": data["img_mask"].to(device)
                }

                image_save = (in_tensor['image'].squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255
                mask_save = (in_tensor['mask'].squeeze().cpu().numpy())
                image_save[:, :, 0] = mask_save * image_save[:, :, 0]
                image_save[:, :, 1] = mask_save * image_save[:, :, 1]
                image_save[:, :, 2] = mask_save * image_save[:, :, 2]
               
                image_save = Image.fromarray(image_save.astype(np.uint8))
               
                image_save.save(osp.join(args.out_dir, "png", f"{data['name']}_image.png"))
                cv2.imwrite(osp.join(args.out_dir, "png", f"{data['name']}_mask.png"),mask_save*255)
    
                # print(data["img_path"])
                
                image_depth=cv2.imread(data["img_path"].replace("images","sapeins_depth"),cv2.IMREAD_GRAYSCALE)/255
                image_depth=cv2.resize(image_depth,(512,512))
                image_depth=image_depth*mask_save
                image_depth = torch.from_numpy(image_depth).unsqueeze(0).unsqueeze(0).to(device).float()
                keypoints_np=sapiens_joints_tokenhmr(data["img_path"].replace("images","sapeins_pose")[:-4]+".json",512,512)
                keypoints_2d=torch.from_numpy(keypoints_np).unsqueeze(0).to(device).float()
                per_data_lst = []

                N_body=1

                smpl_path = f"{args.out_dir}/obj/{data['name']}_smpl_00.obj"


                # remove this line if you change the loop_smpl and obtain different SMPL-X fits
                if osp.exists(smpl_path):

                    smpl_verts_lst = []
                    smpl_faces_lst = []

                    for idx in range(N_body):

                        smpl_obj = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_smpl_{idx:02d}.obj"
                        smpl_mesh = trimesh.load(smpl_obj)
                        smpl_verts = torch.tensor(smpl_mesh.vertices).to(device).float()
                        smpl_faces = torch.tensor(smpl_mesh.faces).to(device).long()
                        smpl_verts_lst.append(smpl_verts)
                        smpl_faces_lst.append(smpl_faces)

                    batch_smpl_verts = torch.stack(smpl_verts_lst)
                    batch_smpl_faces = torch.stack(smpl_faces_lst)

                    # render optimized mesh as normal [-1,1]
                    in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
                        batch_smpl_verts, batch_smpl_faces
                    )

                    with torch.no_grad():
                        in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)

                    in_tensor["smpl_verts"] = batch_smpl_verts * torch.tensor([1., -1., 1.]).to(device)
                    in_tensor["smpl_faces"] = batch_smpl_faces[:, :, [0, 2, 1]]

                else:
                    
                    if args.model_type=="smpl":
                        # The optimizer and variables
                        optimed_pose = data["body_pose"].requires_grad_(False)
                        optimed_trans = data["trans"].requires_grad_(True)
                        optimed_scale = data["scale"].requires_grad_(True)
                        optimed_betas = data["betas"].requires_grad_(False)
                        optimed_orient = data["global_orient"].requires_grad_(True)

                        optimizer_smpl = torch.optim.Adam(
                            [ optimed_trans, optimed_scale,optimed_orient], lr=1e-2, amsgrad=True
                        )
                        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer_smpl,
                            mode="min",
                            factor=0.5,
                            verbose=0,
                            min_lr=1e-4,
                            patience=args.patience,
                        )

                        loop_smpl = tqdm(range(20))

                        for i in loop_smpl:

                            per_loop_lst = []

                            optimizer_smpl.zero_grad()

                            N_body, N_pose = optimed_pose.shape[:2]


                            smpl_output = dataset.smpl_model(betas=optimed_betas, body_pose=optimed_pose, global_orient=optimed_orient, pose2rot=False)
                            smpl_joints = smpl_output.joints
                            smpl_verts = smpl_output.vertices


                            smpl_verts = (smpl_verts + optimed_trans)*optimed_scale

                            # render optimized mesh as normal [-1,1]
                            in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
                                smpl_verts * torch.tensor([1.0, -1.0, -1.0]).to(device),
                                in_tensor["smpl_faces"],
                            )

                            T_mask_F, T_mask_B = dataset.render.get_image(type="mask")



                            with torch.no_grad():
                                in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)

                            diff_F_smpl = torch.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
                            diff_B_smpl = torch.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])

                            # silhouette loss
                            smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)
                            gt_arr = in_tensor["mask"].repeat(1, 1, 2)
                            diff_S = torch.abs(smpl_arr - gt_arr)
                            losses["silhouette"]["value"] = diff_S.mean()

                            img_overlap_path = osp.join(args.out_dir, f"png/{data['name']}_norm_f.png")
                            torchvision.utils.save_image(
                                (((in_tensor["T_normal_F"]).detach().cpu() + 1.0) * 0.5),
                                img_overlap_path
                            )



                            # BUG: PyTorch3D silhouette renderer generates dilated mask
                            bg_value = in_tensor["T_normal_F"][0, 0, 0, 0]
                            smpl_arr_fake = torch.cat([
                                in_tensor["T_normal_F"][:, 0].ne(bg_value).float(),
                                in_tensor["T_normal_B"][:, 0].ne(bg_value).float()
                            ],
                                dim=-1)

                            body_overlap = (gt_arr * smpl_arr_fake.gt(0.0)
                                            ).sum(dim=[1, 2]) / smpl_arr_fake.gt(0.0).sum(dim=[1, 2])
                            body_overlap_mask = (gt_arr * smpl_arr_fake).unsqueeze(1)
                            body_overlap_flag = body_overlap < cfg.body_overlap_thres

                            losses["normal"]["value"] = (
                                                                diff_F_smpl * body_overlap_mask[..., :512] +
                                                                diff_B_smpl * body_overlap_mask[..., 512:]
                                                        ).mean() / 2.0

                            # losses["silhouette"]["weight"] = [0 if flag else 1.0 for flag in body_overlap_flag]


                            # Weighted sum of the losses
                            smpl_loss = 0.0
                            pbar_desc = "Body Fitting -- "
                            for k in ["normal", "silhouette"]:
                                per_loop_loss = (
                                        losses[k]["value"] * torch.tensor(losses[k]["weight"]).to(device)
                                ).mean()
                                pbar_desc += f"{k}: {per_loop_loss:.3f} | "
                                smpl_loss += per_loop_loss
                            pbar_desc += f"Total: {smpl_loss:.3f}"


                            loop_smpl.set_description(pbar_desc)
                            smpl_loss.backward()
                            optimizer_smpl.step()
                            scheduler_smpl.step(smpl_loss)

                        torch.cuda.empty_cache()
                        init_betas= copy.deepcopy(data["betas"])
                        init_poses= copy.deepcopy(data["body_pose"])
                        optimed_pose = data["body_pose"].requires_grad_(True)

                        optimed_scale= copy.deepcopy(optimed_scale).requires_grad_(False)
                        optimed_trans = copy.deepcopy(optimed_trans).requires_grad_(False)
                        optimed_betas = data["betas"].requires_grad_(True)
                        optimed_orient = data["global_orient"].requires_grad_(True)

                        optimizer_smpl = torch.optim.Adam(
                            [optimed_pose, optimed_betas, optimed_orient], lr=1e-2, amsgrad=True
                        )
                        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer_smpl,
                            mode="min",
                            factor=0.5,
                            verbose=0,
                            min_lr=1e-4,
                            patience=args.patience,
                        )

                        # [result_loop_1, result_loop_2, ...]
                        per_data_lst = []

                        N_body, N_pose = optimed_pose.shape[:2]

                        smpl_path = f"{args.out_dir}/obj/{data['name']}_smpl_00.obj"

                        # remove this line if you change the loop_smpl and obtain different SMPL-X fits

                        # smpl optimization
                        loop_smpl = tqdm(range(50))
                        for i in loop_smpl:

                            per_loop_lst = []

                            optimizer_smpl.zero_grad()

                            N_body, N_pose = optimed_pose.shape[:2]


                            
                            smpl_output = dataset.smpl_model(betas=optimed_betas, body_pose=optimed_pose, global_orient=optimed_orient, pose2rot=False)
                            smpl_joints = smpl_output.joints
                            smpl_verts = smpl_output.vertices

                            smpl_verts = (smpl_verts + optimed_trans) * optimed_scale
                            smpl_joints = (smpl_joints + optimed_trans) * optimed_scale * torch.tensor(
                                [1.0, 1.0, -1.0]
                            ).to(device)

                            # torchvision.utils.save_image((T_mask_F.detach().cpu() + 1.0) * 0.5,
                            #                              osp.join(args.out_dir, "png", f"{data['name']}_mask_{i}.png")
                                                        #  )

                            # keypoints=cv2.imread("/media/star/Extreme SSD/write_pic/new_pic17/images/4.png")
                            #
                            smpl_joints_3d_np=smpl_joints.detach().squeeze().cpu().numpy()
                            #
                            keypoints_2d_np=keypoints_2d.detach().squeeze().cpu().numpy()

                            
                            smpl_conf=torch.from_numpy(np.array([1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])).unsqueeze(0).unsqueeze(2).to(device)
                            losses["joint"]["value"]=(smpl_conf * torch.abs(smpl_joints[:,:25,:2]-keypoints_2d)).mean() *10


                            smpl_conf1=np.array([1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
                            keypoints_loss_np=torch.abs(smpl_joints[:,:25,:2]-keypoints_2d).detach().squeeze().cpu().numpy()
                            keypoints_loss_np=np.mean(keypoints_loss_np,axis=1)
                            
                            

                            # smpl_conf=torch.from_numpy(np.array([1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])).unsqueeze(0).unsqueeze(2).to(device)


                            #debug keypoints
                            # for k in range(0,25):
                                # if keypoints_loss_np[k]>0.1:

                                #     cv2.circle(keypoints,  (int((smpl_joints_3d_np[k][0] + 1) * 256), int((smpl_joints_3d_np[k][1] + 1) * 256)),6, (0, 0, int(255)),-1)
                                # else:
                                #     cv2.circle(keypoints,  (int((smpl_joints_3d_np[k][0] + 1) * 256), int((smpl_joints_3d_np[k][1] + 1) * 256)),6, (0, 255, 0),-1)
                                # cv2.circle(keypoints, (
                                # int((keypoints_2d_np[k][0] + 1) * 256), int((keypoints_2d_np[k][1] + 1) * 256)), 6,
                                #            (0, 255, 0), -1)
                            
                            # #     # cv2.putText(keypoints,str(k),(int((smpl_joints_3d_np[k][0]+1)*256),int((smpl_joints_3d_np[k][1]+1)*256)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),1)
                            # #     # cv2.putText(keypoints, str(k), (
                            # #     # int((keypoints_2d_np[k][0] + 1) * 256), int((keypoints_2d_np[k][1] + 1) * 256)),
                            # #     #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                            #     cv2.imwrite(f"/media/star/Extreme SSD/write_pic/new_pic17/pymaf/png/test_lankmark_smpl.jpg",keypoints)
                            


                            # render optimized mesh as normal [-1,1]
                            in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
                                smpl_verts * torch.tensor([1.0, -1.0, -1.0]).to(device),
                                in_tensor["smpl_faces"],
                            )

                            T_mask_F, T_mask_B = dataset.render.get_image(type="mask")

                            depth_F, depth_B = dataset.render_depth(
                                smpl_verts * torch.tensor([1.0, -1.0, -1.0]).to(device),
                                in_tensor["smpl_faces"][:, :, [0, 2, 1]],
                            )
                            depth_F_mask = (depth_F) != -1.0
                            depth_F=depth_F-100
                            
                            depth_F=depth_F*depth_F_mask
                           
                            depth_F=(depth_F-depth_F.min()) / (depth_F.max() - depth_F.min())
                            smpl_depth_F=depth_F*depth_F_mask
                            depth_F2=(smpl_depth_F)*255
                            

                            
                            img_overlap_path = osp.join(args.out_dir,  f"png/{data['name']}_norm_f.png")
                            torchvision.utils.save_image(
                                (((in_tensor["T_normal_F"]).detach().cpu() + 1.0) * 0.5),
                                img_overlap_path
                            )
                            
                            # depthanything

                            


                            image_depth=image_depth*depth_F_mask
                            in_tensor["depth_F"] = smpl_depth_F
                            diff_S_depth = torch.abs((in_tensor["depth_F"] - image_depth))
                            losses["depth"]["value"] = diff_S_depth.mean()

                            with torch.no_grad():
                                in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)

                            diff_F_smpl = torch.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
                            diff_B_smpl = torch.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])

                            # silhouette loss
                            smpl_arr = torch.cat([T_mask_F,T_mask_B], dim=-1)
                            gt_arr = in_tensor["mask"].repeat(1,1,2)
                            diff_S = torch.abs(smpl_arr - gt_arr)
                            losses["silhouette"]["value"] = diff_S.mean()

                            

                           

                            # BUG: PyTorch3D silhouette renderer generates dilated mask
                            bg_value = in_tensor["T_normal_F"][0, 0, 0, 0]
                            smpl_arr_fake = torch.cat([
                                in_tensor["T_normal_F"][:, 0].ne(bg_value).float(),
                                in_tensor["T_normal_B"][:, 0].ne(bg_value).float()
                            ],
                                dim=-1)

                            body_overlap = (gt_arr * smpl_arr_fake.gt(0.0)
                                            ).sum(dim=[1, 2]) / smpl_arr_fake.gt(0.0).sum(dim=[1, 2])
                            body_overlap_mask = (gt_arr * smpl_arr_fake).unsqueeze(1)
                            body_overlap_flag = body_overlap < cfg.body_overlap_thres

                            losses["normal"]["value"] = (
                                                                diff_F_smpl * body_overlap_mask[..., :512] +
                                                                diff_B_smpl * body_overlap_mask[..., 512:]
                                                        ).mean() / 2.0

                            # losses["silhouette"]["weight"] = [0 if flag else 1.0 for flag in body_overlap_flag]


                            # Weighted sum of the losses
                            smpl_loss = 0.0
                            pbar_desc = "Body Fitting -- "
                            for k in ["normal","joint","depth"]:
                                per_loop_loss = (
                                        losses[k]["value"] * torch.tensor(losses[k]["weight"]).to(device)
                                ).mean()
                                pbar_desc += f"{k}: {per_loop_loss:.3f} | "
                                smpl_loss += per_loop_loss
                            pbar_desc += f"Total: {smpl_loss:.3f}"

                            loop_smpl.set_description(pbar_desc)

                            # save intermediate results
                            if (i == args.loop_smpl - 1) :
                                per_loop_lst.extend([
                                    in_tensor["image"],
                                    in_tensor["T_normal_F"],
                                    in_tensor["normal_F"],
                                    diff_S[:, :, :512].unsqueeze(1).repeat(1, 3, 1, 1),
                                ])
                                per_loop_lst.extend([
                                    in_tensor["image"],
                                    in_tensor["T_normal_B"],
                                    in_tensor["normal_B"],
                                    diff_S[:, :, 512:].unsqueeze(1).repeat(1, 3, 1, 1),
                                ])
                                per_data_lst.append(
                                    get_optim_grid_image(per_loop_lst, None, nrow=N_body * 2, type="smpl")
                                )

                            smpl_loss.backward()
                            optimizer_smpl.step()
                            scheduler_smpl.step(smpl_loss)

                        in_tensor["smpl_verts"] = smpl_verts * torch.tensor([1.0, 1.0, -1.0]).to(device)
                        in_tensor["smpl_faces"] = in_tensor["smpl_faces"][:, :, [0, 2, 1]]

                        




                    smpl_obj_lst = []

                    for idx in range(N_body):
                        smpl_obj_verts=in_tensor["smpl_verts"].detach().cpu()[idx] * torch.tensor([1.0, -1.0, 1.0])
                        smpl_obj_verts[:,2]=smpl_obj_verts[:,2]*2
                        smpl_obj = trimesh.Trimesh(
                            smpl_obj_verts,
                            in_tensor["smpl_faces"].detach().cpu()[0][:, [0, 2, 1]],
                            process=False,
                            maintains_order=True,
                        )

                        smpl_obj_path = f"{args.out_dir}/obj/{data['name']}_smpl_{idx:02d}_60.obj"

                        smpl_obj.export(smpl_obj_path)
                        pose=optimed_pose.detach().cpu().numpy()
                        ori_root=optimed_orient.detach().cpu().numpy()
                        pose = np.concatenate((ori_root, pose), axis=1)

                        rotmat_host = (pose)[0]
                        theta_host = []
                        for r in rotmat_host:
                            theta_host.append(cv2.Rodrigues(r)[0])
                        pose_72 = np.asarray(theta_host).reshape((1, -1))
                        smpl_param_name = f"{args.out_dir}/obj/{data['name']}_smpl_{idx:02d}.pkl"
                        with open(smpl_param_name, 'wb') as fp:
                            pkl.dump({'betas': optimed_betas.detach().cpu().numpy(),
                                      'pose1': optimed_pose.detach().cpu().numpy(),
                                      "ori_root1":optimed_orient.detach().cpu().numpy(),
                                      'pose':pose_72,
                                      "scale":optimed_scale.detach().cpu().numpy(),
                                      "trans":optimed_trans.detach().cpu().numpy().squeeze(),
                                      },
                                     fp)

                        #
                        smpl_obj_path2= f"{args.out_dir}/obj/{data['name']}_smpl_{idx:02d}2.obj"
                        smpl = SMPL(
                            '/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/data/smpl_related/model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').to(
                            device)


                        pose2 = torch.Tensor(pose_72).float().to(device)

                        gt_v = smpl(pose2, optimed_betas)
                        trimesh.Trimesh(
                            ((gt_v.squeeze().detach().cpu().numpy() +optimed_trans.detach().cpu().numpy().squeeze())*optimed_scale.detach().cpu().numpy()) * np.array([1, -1, -1]),
                            in_tensor["smpl_faces"].detach().cpu()[0][:, [0, 2, 1]]).export(
                            smpl_obj_path2)



