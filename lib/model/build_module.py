from lib.model.unet_2d import UNet2D
from lib.model.unet_3d import UNet3D, UNet3DLight
from lib.model.corrector import Corrector
from lib.data.dataset_utils import ConcatI2IDataset, ConcatS2SDataset
# from lib.config.cfg import cfg
import torch


def build_correction_model(cfg):
    if cfg.model_type == "UNet2D":
        unet = UNet2D(*cfg.model_args.unet2d.values())
        model = Corrector(unet).to(cfg.device)
    elif cfg.model_type == "UNet3D":
        unet = UNet3D(*cfg.model_args.unet3d.values())
        model = Corrector(unet).to(cfg.device)
    elif cfg.model_type == "UNet3DLight":
        unet = UNet3DLight(*cfg.model_args.unet3d.values())
        model = Corrector(unet).to(cfg.device)
    else:
        raise NotImplementedError
    return model


def build_dataset(*datasets, cfg):
    if cfg.model_type == "UNet2D":
        dataset = ConcatI2IDataset(*datasets)
        print(dataset, 1)
    elif cfg.model_type == "UNet3D":
        dataset = ConcatS2SDataset(*datasets, **cfg.s2s)
        print(dataset, 2)
    elif cfg.model_type == "UNet3DLight":
        dataset = ConcatS2SDataset(*datasets, **cfg.s2s)
        print(dataset, 3)
    else:
        raise NotImplementedError
    return dataset
