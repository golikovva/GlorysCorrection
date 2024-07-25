from lib.model.unet_2d import UNet2D
from lib.model.unet_3d import UNet3D, UNet3DLight
from lib.model.partial_unet_3d import PartUNet3DLight
from lib.model.corrector import Corrector
from lib.model.corr_accumulator import AccumCorrector
from lib.data.dataset_utils import ConcatI2IDataset, ConcatS2SDataset

import torch


def build_correction_model(cfg):
    if cfg.model_type == "UNet2D":
        unet = UNet2D(*cfg.model_args.unet2d.values())
        model = Corrector(unet).to(cfg.device)
    elif cfg.model_type == "UNet3D":
        unet = UNet3D(*cfg.model_args.unet3d.values())
        model = Corrector(unet, n_classes=cfg.model_args.unet2d.n_classes, classes_dim=-3).to(cfg.device)
    elif cfg.model_type == "UNet3DLight":
        print(cfg.model_args.unet3d.n_channels)
        unet = UNet3DLight(*cfg.model_args.unet3d.values())
        model = Corrector(unet, n_classes=cfg.model_args.unet3d.n_classes, classes_dim=-4).to(cfg.device)
    elif cfg.model_type == "partunet3d":
        print(cfg.model_args.unet3d.n_channels)
        unet = PartUNet3DLight(*cfg.model_args.partunet3d.values())
        model = Corrector(unet, n_classes=cfg.model_args.partunet3d.n_classes, classes_dim=-4).to(cfg.device)
    else:
        raise NotImplementedError
    return model


def build_dataset(*datasets, cfg):
    if cfg.model_type == "UNet2D":
        dataset = ConcatI2IDataset(*datasets)
    elif cfg.model_type == "UNet3D":
        dataset = ConcatS2SDataset(*datasets, **cfg.s2s)
    elif cfg.model_type == "UNet3DLight":
        dataset = ConcatS2SDataset(*datasets, **cfg.s2s)
    elif cfg.model_type == "partunet3d":
        dataset = ConcatS2SDataset(*datasets, **cfg.s2s, return_mask=True)
    else:
        raise NotImplementedError
    return dataset


def build_inference_correction_model(cfg):
    if cfg['model_type'] == "spatiotemporal_baseline":
        model = AccumCorrector(cfg['model_weights'])
    elif cfg['model_type'] == "unet2d":
        unet = UNet2D(34, 34)
        model = Corrector(unet).to(cfg['device'])
        state_dict = torch.load(cfg['model_weights'])
        model.load_state_dict(state_dict)
    elif cfg['model_type'] == "unet3d":
        unet = UNet3D(4, 1, bilinear=False, factor=8)
        model = Corrector(unet, n_classes=1, classes_dim=-3).to(cfg['device'])
        state_dict = torch.load(cfg['model_weights'])
        model.load_state_dict(state_dict)
    elif cfg['model_type'] == "unet3dlight":
        unet = UNet3DLight(4, 1, bilinear=False, factor=8)
        model = Corrector(unet, n_classes=1, classes_dim=-4).to(cfg['device'])
        state_dict = torch.load(cfg['model_weights'])
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError
    model.eval()
    return model
