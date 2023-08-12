import pytorch_lightning as pl
import torch
from loguru import logger

from model.full_model import GeoFormer
from model.loftr_src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from model.loftr_src.losses.loftr_loss import LoFTRLoss, GeoLoss
from model.loftr_src.optimizers import build_optimizer, build_scheduler
from model.loftr_src.utils.metrics import (
    compute_symmetrical_epipolar_errors
)
from model.loftr_src.utils.misc import lower_config
from model.loftr_src.utils.plotting import make_matching_figures
from model.loftr_src.utils.profiler import PassThroughProfiler


class PL_H_GeoFormer(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):

        super().__init__()
        # Misc
        self.config = config  # full loftr_config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['loftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = GeoFormer(loftr_config=_config['loftr'])
        self.loss = GeoLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir
        
    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)
        
        with self.profiler.profile("GeoFormer"):
            self.matcher(batch)
        
        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)
            
        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        if batch_idx % 100 == 0:
            dt = {'state_dict': self.matcher.state_dict()}
            if self.global_rank == 0:
                torch.save(dt, f'/data3/ljz/matching/tmp_save/tmp{batch_idx}.ckpt')

        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

            # net-params
            if self.config.LOFTR.MATCH_COARSE.MATCH_TYPE == 'sinkhorn':
                self.logger.experiment.add_scalar(
                    f'skh_bin_score', self.matcher.coarse_matching.bin_score.clone().detach().cpu().data, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                for k, v in figures.items():
                    self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)
        torch.cuda.empty_cache()
        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)
    
    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        self.log('val_loss', batch['loss'])
    #
    # def validation_epoch_end(self, outputs):
    #     return

