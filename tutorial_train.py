from dataclasses import dataclass

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import FashionDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

@dataclass
class Config:

    # training
    resume_path: str
    model_config: str
    batch_size: int
    logger_freq: int
    learning_rate: float
    sd_locked: bool
    only_mid_control: bool

    # data
    image_dir: str
    df_path: str
    attributes_path: str
    width: int
    height: int


def train_controlnet(opt: Config):

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(opt.model_config).cpu()
    model.load_state_dict(load_state_dict(opt.resume_path, location='cpu'))
    model.learning_rate = opt.learning_rate
    model.sd_locked = opt.sd_locked
    model.only_mid_control = opt.only_mid_control

    # Misc
    dataset = FashionDataset(opt)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=opt.logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

    # Train!
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    # for training version 1
    # resume_path = './models/control_sd21_ini.ckpt',
    # model_config = './models/cldm_v21.yaml',

    config = Config(
        resume_path='./models/control_sd21_ini.ckpt',
        model_config='./models/cldm_v21.yaml',
        batch_size=32,
        logger_freq=300,
        learning_rate=1e-5,
        sd_locked=True,
        only_mid_control=False,
        image_dir='./training/iMaterialist(Fashion)/train',
        df_path='./training/iMaterialist(Fashion)/train.csv',
        attributes_path='./training/iMaterialist(Fashion)/label_descriptions.json',
        width=512,
        height=512,
    )

    train_controlnet(config)
