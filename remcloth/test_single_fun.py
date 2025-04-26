import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os.path as osp
from remcloth.src.models.modnet import MODNet
import numpy as np
from PIL import Image
# define image to tensor transform
im_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)


def re_colth(image_path,output_path):
    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=True)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(
            "/media/star/Extreme SSD/code/VS/remcloth/pretrained/modnet_photographic_portrait_matting.ckpt")
    else:
        weights = torch.load(
            "/media/star/Extreme SSD/code/VS/remcloth/pretrained/modnet_photographic_portrait_matting.ckpt",
            map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()
    ref_size = 512

    print(image_path)
    path,im_name = os.path.split(image_path)
    # read image
    im = Image.open(image_path)

    # unify image channels to 3
    im = np.asarray(im)
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
    matte_name = im_name.split('.')[0] + '_mask.png'
    # Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(
    #     os.path.join(output_path, matte_name))
    return (matte * 255).astype('uint8')


def re_colth_np(image_np):
    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=True)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(
            "/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/data/pretrained/model_epoch_200.pth")
    else:
        weights = torch.load(
            "/media/bbnc/FE345AE3345A9F09/loose_cloth/clothHMR/data/pretrained/model_epoch_200.pth",
            map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()
    ref_size = 512

    # unify image channels to 3
    im = np.asarray(image_np)
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

    im = np.asarray(image_np)
    im_rembg = im * (matte[:, :, None].repeat(3, axis=2))
    im_rembg = im_rembg.astype(np.uint8)
    # print(im_rembg.shape)
    # Image.fromarray(im_rembg).save(
    #     os.path.join("/media/star/Extreme SSD/code/VS/img3.png"))

    matte = matte*255
    trimap = np.zeros_like(matte)
    trimap[matte>10]=1
    return im_rembg,trimap.astype('uint8')

