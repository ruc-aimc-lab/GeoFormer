import math
import os
import sys

from lightning_homo_geoformer import PL_H_GeoFormer
from model.loftr_src.config.default import get_cfg_defaults
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from model.loftr_src.lightning.data import HomoDataModule
# from data_depth import MultiSceneDataModule
from model.loftr_src.utils.misc import get_rank_zero_only_logger, setup_gpus
from model.loftr_src.utils.profiler import build_profiler

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    # init a costum parser which will be added into pl.MyTrainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data_cfg_path', type=str, help='data loftr_config path', default='../train_config/homo_trainval_640.py')
    parser.add_argument(
        '--loftr_cfg_path', type=str, help='loftr loftr_config path', default='../train_config/loftr_ds_dense.py')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=8)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    import os
    # parse arguments
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    args = parse_args()
    args.num_nodes = 1
    args.gpus = 8
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    default_config = get_cfg_defaults()
    # model_config = get_cfg_model()
    default_config.merge_from_file(args.loftr_cfg_path)
    default_config.merge_from_file(args.data_cfg_path)
    # pl.seed_everything(default_config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    default_config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    default_config.TRAINER.TRUE_BATCH_SIZE = default_config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = default_config.TRAINER.TRUE_BATCH_SIZE / default_config.TRAINER.CANONICAL_BS
    default_config.TRAINER.SCALING = _scaling
    default_config.TRAINER.TRUE_LR = default_config.TRAINER.CANONICAL_LR * _scaling
    default_config.TRAINER.WARMUP_STEP = math.floor(default_config.TRAINER.WARMUP_STEP / _scaling)
    # lightning module
    profiler = build_profiler(args.profiler_name)

    model = PL_H_GeoFormer(default_config,
                         profiler=profiler)


    loguru_logger.info(f"LightningModule initialized!")

    # lightning data
    data_module = HomoDataModule(args, default_config)
    loguru_logger.info(f"DataModule initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    print(ckpt_dir)

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(monitor='val_loss', verbose=True, save_top_k=5, mode='min',
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{val_loss:.3f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning MyTrainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(find_unused_parameters=True,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=default_config.TRAINER.WORLD_SIZE > 0),
        gradient_clip_val=default_config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=default_config.TRAINER.WORLD_SIZE > 0,
        weights_summary='full',
        profiler=profiler)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()