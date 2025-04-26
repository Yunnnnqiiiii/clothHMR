import os
import sys
import argparse

import cv2
import numpy as np
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os.path as osp
from src.models.modnet import MODNet


if __name__ == '__main__':


    # img_path=os.path.join("/media/star/Extreme SSD/remobe_recloth/images")
    # print(img_path)
    #
    # output_path=os.path.join("/media/star/Extreme SSD/remobe_recloth/")
    # trimap_output_path=os.path.join("/media/star/Extreme SSD/cloth4d/CLOTH4Dsub_8views_part3/s333/00_000060_uv","T_normal_F_trimap2")
    # dataset_path = "/media/star/Extreme SSD/EDBM/mask/"
    # /media/star/Extreme SSD/EDBM/images_crop/P1/14_outdoor_climb/images
    # for iuput_path1 in sorted(os.listdir(dataset_path))[4:]:

    #     for inupt_path2 in sorted(os.listdir(os.path.join(dataset_path, iuput_path1))):
    #         print(iuput_path1)
            # input_path = os.path.join(dataset_path, iuput_path1, inupt_path2, "images")
            # output_path = input_path.replace("mask", "rmcloth")
            input_path="/media/star/Extreme SSD/write_pic/new_pic18/test"
            output_path="/media/star/Extreme SSD/write_pic/new_pic18/rmcloth4"
            parser = argparse.ArgumentParser()
            parser.add_argument('--input-path', default=input_path,type=str, help='path of input images')
            parser.add_argument('--output-path',default=output_path, type=str, help='path of output images')
            # parser.add_argument('--trimap_output_path',
            #                     default=trimap_output_path,
            #                     type=str, help='path of output images')

            parser.add_argument('--ckpt-path', default="/media/star/Extreme SSD/code/MODNet-master/pretrained/human-seg/all_data/model_epoch_200.pth",type=str, help='path of pre-trained MODNet')
            args = parser.parse_args()

            # check input arguments
            if not os.path.exists(args.input_path):
                print('Cannot find input path: {0}'.format(args.input_path))
                #exit()
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path,exist_ok=True)
                # os.mkdir(args.output_path)
                print('Cannot find output path: {0}'.format(args.output_path))
            # if not os.path.exists(args.trimap_output_path):
            #     os.makedirs(args.trimap_output_path, exist_ok=True)
            #     # os.mkdir(args.output_path)
            #     print('Cannot find output path: {0}'.format(args.output_path))

            if not os.path.exists(args.ckpt_path):
                print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
                #exit()

            # define hyper-parameters
            ref_size = 512

            # define image to tensor transform
            im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )

            # create MODNet and load the pre-trained ckpt
            modnet = MODNet(backbone_pretrained=True)
            modnet = nn.DataParallel(modnet)

            if torch.cuda.is_available():
                modnet = modnet.cuda()
                weights = torch.load(args.ckpt_path)
            else:
                weights = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
            modnet.load_state_dict(weights)
            modnet.eval()

            # inference images
            im_names = os.listdir(args.input_path)
            for im_name in im_names:
                print('Process image: {0}'.format(im_name))

                # read image
                im2 = Image.open(os.path.join(args.input_path, im_name))

                # unify image channels to 3
                im = np.asarray(im2)
                if len(im.shape) == 2:
                    im = im[:, :, None]
                if im.shape[2] == 1:
                    im = np.repeat(im, 3, axis=2)
                elif im.shape[2] == 4:
                    im = im[:, :, 0:3]

                # convert image to PyTorch tensor
                im = Image.fromarray(im)
                im = im_transform(im)

                # add mini-batch dim
                im = im[None, :, :, :]

                # resize image for input
                im_b, im_c, im_h, im_w = im.shape
                if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
                    if im_w >= im_h:
                        im_rh = ref_size
                        im_rw = int(im_w / im_h * ref_size)
                    elif im_w < im_h:
                        im_rw = ref_size
                        im_rh = int(im_h / im_w * ref_size)
                else:
                    im_rh = im_h
                    im_rw = im_w

                im_rw = im_rw - im_rw % 32
                im_rh = im_rh - im_rh % 32
                im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

                # inference
                _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

                # resize and save matte
                matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
                matte = matte[0][0].data.cpu().numpy()
                matte_name = im_name.split('.')[0] + '.png'
                # black = np.zeros_like(matte)
                # print((matte*255)>1)
                # black[(matte*255)>35]=1
                Image.fromarray(((matte*255).astype('uint8')), mode='L').save(os.path.join(args.output_path, matte_name))
                # im = np.asarray(im2)
                # white=np.zeros_like(im)
                # # print(matte[:, :, None].repeat(3, axis=2).shape)
                # im_rembg = im[:,:,:3] * (matte[:, :, None].repeat(3, axis=2))+white[:,:,:3]*((1-matte[:, :, None]).repeat(3, axis=2))
                # im_rembg = im_rembg.astype(np.uint8)



                # Image.fromarray(im_rembg).save(
                #     os.path.join(args.output_path, matte_name))





                # matte = matte * 255
                # fg=(matte>235).astype(np.uint8)
                # bg=(matte<35).astype(np.uint8)
                # unknown_region=((matte>35)&(matte<235)).astype(np.uint8)
                # trimap=np.zeros_like(matte)
                # trimap[fg==1]=255
                # trimap[bg==1]=0
                # trimap[unknown_region==1]=128
                #
                # trimap_name = im_name.split('.')[0] + '.png'
                # Image.fromarray(((trimap).astype('uint8')), mode='L').save(os.path.join(args.trimap_output_path, trimap_name))


