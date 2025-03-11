import argparse
import datetime
import logging
import os
import time
import traceback
from typing import Union

import dask
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    RandomSampler,
)

from config import TrainConfig
from constants import BOUND_VARS_MAP, PROG_VARS_MAP, TensorMap, construct_metadata
from datasets import InferenceDataset, InferenceDatasets, TrainDataset
from models.samudra import Samudra
from stepper import Stepper, TrainOutput, ValOutput
from utils.data import Normalize, extract_wet_mask, get_inference_steps, validate_data
from utils.device import get_device, using_gpu
from utils.distributed import (
    all_reduce_mean,
    get_world_size,
    init_distributed_mode,
    is_main_process,
    set_seed,
)
from utils.logging import MetricLogger, SmoothedValue, handle_logging, handle_warnings
from utils.train import (
    CheckpointPaths,
    collate_inference_data,
    collate_train_data,
    decomposed_mse,
)


class Trainer:
    def __init__(self, cfg) -> None:
        if not using_gpu():
            logging.info("No GPU available, using CPU")
            cfg.distributed.enabled = False

        self.device = get_device()

        # Adjust workers and memory pinning based on device
        if not using_gpu():
            cfg.data.num_workers = 0  # Disable multi-processing on CPU
            cfg.pin_mem = False
        elif cfg.disk_mode:
            cfg.data.num_workers = torch.cuda.device_count() * cfg.data.num_workers
            cfg.pin_mem = True

        # Distributed mode
        init_distributed_mode(cfg.distributed)
        dask.config.set(scheduler="synchronous")

        # Set seeds
        set_seed(cfg.experiment.rand_seed)

        # Getting prognostic and boundary variables
        self.prognostic_vars = PROG_VARS_MAP[cfg.experiment.prognostic_vars_key]
        self.boundary_vars = BOUND_VARS_MAP[cfg.experiment.boundary_vars_key]

        self.levels = 19

        self.str_prog_vars = ", ".join([i for i in self.prognostic_vars])
        self.str_bound_vars = ", ".join([i for i in self.boundary_vars])

        logging.info(f"Prognostic variables: {self.str_prog_vars}")
        logging.info(f"Boundary variables: {self.str_bound_vars}")
        logging.info(f"Levels: {self.levels}")

        self.N_bound = len(self.boundary_vars)
        self.N_prog = len(self.prognostic_vars)

        self.num_in = int((cfg.data.hist + 1) * self.N_prog + self.N_bound)
        self.num_out = int((cfg.data.hist + 1) * self.N_prog)

        self.tensor_map = TensorMap.init_instance(
            cfg.experiment.prognostic_vars_key, cfg.experiment.boundary_vars_key
        )

        logging.info(
            f"Number of inputs: hist * prognostic_vars + boundary_vars = {self.num_in}"
        )
        logging.info(f"Number of outputs: hist * prognostic_vars = {self.num_out}")

        assert isinstance(cfg.data_stride, list)
        assert isinstance(cfg.steps, list)
        assert isinstance(cfg.step_transition, list)
        assert len(cfg.step_transition) == len(cfg.steps) - 1

        # Dataloaders
        logging.info(f"Loading data")
        self.data_dir = cfg.experiment.data_dir
        self.data_path = cfg.data.data_path
        self.data_means_path = cfg.data.data_means_path
        self.data_stds_path = cfg.data.data_stds_path

        data = xr.open_zarr(os.path.join(self.data_dir, self.data_path), chunks={})

        data_mean = xr.open_dataset(
            os.path.join(self.data_dir, self.data_means_path),
            engine="zarr",
            chunks={},
        )
        data_std = xr.open_dataset(
            os.path.join(self.data_dir, self.data_stds_path),
            engine="zarr",
            chunks={},
        )

        self.data, self.data_mean, self.data_std = validate_data(
            data, data_mean, data_std
        )

        self.metadata = construct_metadata(self.data)
        self.wet, self.wet_surface = extract_wet_mask(
            self.data, self.prognostic_vars, cfg.data.hist
        )
        wet_without_hist, _ = extract_wet_mask(self.data, self.prognostic_vars, 0)

        self.normalize = Normalize.init_instance(
            self.data_mean,
            self.data_std,
            self.prognostic_vars,
            self.boundary_vars,
            wet_without_hist,
        )

        # Model
        logging.info(f"Instantiating model {cfg.experiment.network}")
        if "samudra" == cfg.experiment.network:
            if cfg.unet.ch_width[0] != self.num_in:
                logging.info(
                    f"NOTE: Changing input channels to match data "
                    f"{cfg.unet.ch_width[0]}->{self.num_in}"
                )
                cfg.unet.ch_width[0] = self.num_in
            if cfg.unet.n_out != self.num_out:
                logging.info(
                    f"NOTE: Changing output channels to match data "
                    f"{cfg.unet.n_out}->{self.num_out}"
                )
                cfg.unet.n_out = self.num_out
            model = Samudra(
                cfg.unet, hist=cfg.data.hist, wet=self.wet.to(self.device)
            ).to(self.device)
        else:
            raise NotImplementedError

        self.model = model
        self.nets_dir = cfg.experiment.nets_dir
        self.network = cfg.experiment.network

        # Loss function
        logging.info("Loss = mse")
        if cfg.loss == "mse":
            self.loss_fn = decomposed_mse
        else:
            raise NotImplementedError

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        # Scheduler
        self.scheduler = None
        if cfg.scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs
            )

        # Modify DDP setup based on device
        if using_gpu():
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[cfg.distributed.gpu]
            )

        # Training
        self.epochs = cfg.epochs
        self.hist = cfg.data.hist
        self.steps = cfg.steps
        self.step_transition = cfg.step_transition
        self.save_freq = cfg.save_freq
        self.output_dir = cfg.experiment.output_dir
        self.network = cfg.experiment.network
        self.debug = cfg.debug
        self.data_stride = cfg.data_stride
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.data.num_workers
        self.pin_mem = cfg.pin_mem
        self.train_times = cfg.train
        self.val_times = cfg.val
        self.inference_times = cfg.inference
        self.inference_epochs = cfg.inference_epochs
        self.time_delta = cfg.data.time_delta
        self.num_batches_seen = 0
        self.start_epoch = 0
        self.ckpt_paths = CheckpointPaths(self.nets_dir)

        assert self.tensor_map is not None

        self.init_inference_stores()

        # Add type annotations for samplers
        self.train_sampler: Union[DistributedSampler, RandomSampler]
        self.val_sampler: Union[DistributedSampler, RandomSampler]
        self.inference_sampler: Union[DistributedSampler, RandomSampler]

        # Add type annotations for loaders
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.inference_loader: DataLoader

    def init_inference_stores(self):
        # Determine number of processes based on device
        if using_gpu():
            num_splits = get_world_size()
        else:
            num_splits = 1

        # Create datasets
        inference_datasets = []
        num_steps_inf_set = []
        for i in range(num_splits):
            num_time_steps = get_inference_steps(
                self.inference_times[i],
                time_delta=self.time_delta,
                hist=self.hist,
            )
            inference_data = self.data.sel(
                time=slice(
                    self.inference_times[i].start_time,
                    self.inference_times[i].end_time,
                )
            )
            inference_dataset = InferenceDataset(
                inference_data,
                self.prognostic_vars,
                self.boundary_vars,
                self.wet,
                self.wet_surface,
                self.hist,
            )

            inference_datasets.append(inference_dataset)
            num_steps_inf_set.append(num_time_steps)

        inference_data_combined: Dataset = InferenceDatasets(
            inference_datasets, num_steps_inf_set
        )

        if using_gpu():
            self.inference_sampler = DistributedSampler(
                inference_data_combined, shuffle=True
            )
        else:
            self.inference_sampler = RandomSampler(inference_data_combined)

        # Create data loaders
        self.inference_loader = DataLoader(
            inference_data_combined,
            batch_size=1,
            sampler=self.inference_sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_inference_data,
        )

    def run(self) -> None:
        self.best_val_loss = torch.tensor(1e8)
        self.best_inf_loss = torch.tensor(1e8)

        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Iterative step training
            if epoch == self.start_epoch or epoch in self.step_transition:
                cur_step = self.get_current_step(epoch)
                self.init_data_loaders(cur_step)

            if isinstance(self.train_sampler, DistributedSampler):
                self.train_sampler.set_epoch(epoch)
            if isinstance(self.val_sampler, DistributedSampler):
                self.val_sampler.set_epoch(epoch)

            start_epoch_train_time = time.time()
            train_loss = self.train_one_epoch(epoch)
            end_epoch_train_time = time.time()
            v_loss = self.validate_one_epoch(epoch)
            end_epoch_val_time = time.time()

            if -1 in self.inference_epochs or epoch in self.inference_epochs:
                inf_loss = self.inference_one_epoch(epoch)
                end_epoch_inf_time = time.time()
            else:
                inf_loss = None
                end_epoch_inf_time = None

            logging.info(f"Achieved Train Loss = {train_loss:.3f}")
            logging.info(f"Achieved Validation Loss = {v_loss:.3f}")
            if inf_loss is not None:
                logging.info(f"Achieved Inference Loss = {inf_loss:.3f}")

            if is_main_process():
                self.save_all_checkpoints(epoch, v_loss, inf_loss)

            time_elapsed = time.time() - start_epoch_train_time
            logging.info(
                f"Train time: {end_epoch_train_time - start_epoch_train_time:.3f}s"
            )
            logging.info(
                f"Validation time: {end_epoch_val_time - end_epoch_train_time:.3f}s"
            )
            if end_epoch_inf_time is not None:
                logging.info(
                    f"Inference time: {end_epoch_inf_time - end_epoch_val_time:.3f}s"
                )
            logging.info(f"Total time: {time_elapsed:.3f}s")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(f"Training time {total_time_str}")

    def train_one_epoch(self, epoch):
        self.model.train(True)
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = "Training Epoch: [{}]".format(epoch)
        # iters = len(self.train_loader)
        total_train_loss = 0.0
        for data_iter_step, data in enumerate(
            metric_logger.log_every(self.train_loader, 1, header)
        ):
            if self.debug and (data_iter_step + 1) % 5 == 0:
                break

            self.optimizer.zero_grad()
            data.to(self.device)
            TO: TrainOutput = Stepper.train_step(self.model, data, self.loss_fn)
            TO.loss.backward()

            self.num_batches_seen += 1

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            lr = (
                self.optimizer.param_groups[-1]["lr"]
                if self.scheduler is None
                else self.scheduler.get_last_lr()[0]
            )

            with torch.no_grad():
                # Reduce losses
                loss_value_reduce = all_reduce_mean(TO.loss.detach())

            metric_logger.update(loss=loss_value_reduce.item())
            metric_logger.update(lr=lr)

            total_train_loss += loss_value_reduce.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return total_train_loss / len(self.train_loader)

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        self.model.eval()

        metric_logger = MetricLogger(delimiter="  ")
        header = "One-Step Validation Epoch: [{}]".format(epoch)

        total_val_loss = 0.0
        for data_iter_step, data in enumerate(
            metric_logger.log_every(self.val_loader, 1, header)
        ):
            if self.debug and (data_iter_step + 1) % 5 == 0:
                break

            data.to(self.device)
            VO: ValOutput = Stepper.validate_step(self.model, data, self.loss_fn)
            metric_logger.update(loss=VO.loss)

            total_val_loss += VO.loss.item()

        return total_val_loss / len(self.val_loader)

    @torch.no_grad()
    def inference_one_epoch(self, epoch):
        self.model.eval()

        total_inf_loss = 0
        for _, (inference_dataset, num_steps) in enumerate(self.inference_loader):
            inference_loss = Stepper.inference(
                model=self.model.module if using_gpu() else self.model,
                dataset=inference_dataset,
                epoch=epoch,
                num_model_steps_forward=num_steps,
                loss_fn=self.loss_fn,
            )
            total_inf_loss += inference_loss

        return total_inf_loss / len(self.inference_loader)

    def get_current_step(self, epoch):
        """Determine the current step based on the epoch and transition points.

        Args:
            epoch (int): Current epoch number

        Returns:
            tuple: (current_step, current_step_idx)
        """
        if epoch == self.start_epoch:
            # Find initial step based on start epoch
            cur_step = None
            cur_step_idx = None
            for i, epoch_to_transition in enumerate(self.step_transition):
                if epoch <= epoch_to_transition:
                    cur_step = self.steps[i]
                    cur_step_idx = i
                    break
            if cur_step is None:
                cur_step = self.steps[-1]
                cur_step_idx = len(self.steps) - 1
            logging.info(f"Starting training at step {cur_step}")
        else:
            # Transition to next step
            cur_step_idx = next(
                i for i, e in enumerate(self.step_transition) if e == epoch
            )
            cur_step_idx += 1
            cur_step = self.steps[cur_step_idx]
            logging.info(f"Transitioning to step {cur_step}")

        return cur_step

    def init_data_loaders(self, cur_step: int) -> None:
        """Initialize training and validation data loaders.

        Args:
            cur_step: Current training step size
        """
        # Create datasets
        train_data: Dataset = ConcatDataset(
            [
                TrainDataset(
                    self.data.sel(
                        time=slice(
                            self.train_times.start_time,
                            self.train_times.end_time,
                        )
                    ),
                    self.prognostic_vars,
                    self.boundary_vars,
                    self.wet,
                    self.wet_surface,
                    self.hist,
                    cur_step,
                    stride,
                )
                for stride in self.data_stride
            ]
        )

        val_data: Dataset = ConcatDataset(
            [
                TrainDataset(
                    self.data.sel(
                        time=slice(
                            self.val_times.start_time,
                            self.val_times.end_time,
                        )
                    ),
                    self.prognostic_vars,
                    self.boundary_vars,
                    self.wet,
                    self.wet_surface,
                    self.hist,
                    1,  # current_step set to 1 for validation
                    stride,
                )
                for stride in self.data_stride
            ]
        )

        logging.info("Instantiating torch loaders")

        if using_gpu():
            self.train_sampler = DistributedSampler(train_data, shuffle=True)
            self.val_sampler = DistributedSampler(val_data, shuffle=False)
        else:
            self.train_sampler = RandomSampler(train_data)
            self.val_sampler = RandomSampler(val_data)

        # Create data loaders
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=True,
            collate_fn=collate_train_data,
        )

        self.val_loader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=False,
            collate_fn=collate_train_data,
        )

    def save_all_checkpoints(self, epoch, v_loss, inf_loss):
        save_best_checkpoint = False
        if v_loss <= self.best_val_loss:
            logging.info(
                f"Epoch validation loss ({v_loss}) is lower than "
                f"previous best validation loss ({self.best_val_loss})."
            )
            logging.info(
                "Saving lowest validation loss checkpoint to "
                f"{self.ckpt_paths.best_validation_checkpoint_path}"
            )
            self.best_val_loss = v_loss
            save_best_checkpoint = True  # wait until inference error is updated
        if inf_loss is not None and (inf_loss <= self.best_inf_loss):
            logging.info(
                f"Epoch inference error ({inf_loss}) is lower than "
                f"previous best inference error ({self.best_inf_loss})."
            )
            logging.info(
                "Saving lowest inference error checkpoint to "
                f"{self.ckpt_paths.best_inference_checkpoint_path}"
            )
            self.best_inf_loss = inf_loss
            self.save_checkpoint(epoch, self.ckpt_paths.best_inference_checkpoint_path)
        if save_best_checkpoint:
            self.save_checkpoint(epoch, self.ckpt_paths.best_validation_checkpoint_path)

        logging.info(
            f"Saving latest checkpoint to {self.ckpt_paths.latest_checkpoint_path}"
        )
        self.save_checkpoint(epoch, self.ckpt_paths.latest_checkpoint_path)
        if epoch > 0 and epoch % self.save_freq == 0:
            self.save_checkpoint(
                epoch, self.ckpt_paths.latest_checkpoint_path_with_epoch(epoch)
            )

    def save_checkpoint(self, epoch, checkpoint_path):
        checkpoint = {
            "model": self.model.module.state_dict()
            if using_gpu()
            else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "best_inf_loss": self.best_inf_loss,
        }
        if self.scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--subname", type=str, required=False, help="Subname for the run", default=""
    )
    args = parser.parse_args()

    overrides = {}
    if args.subname:
        overrides["sub_name"] = args.subname

    # Load config from YAML
    cfg = TrainConfig.from_yaml(args.config, overrides)

    # Check dirs
    if not os.path.exists(cfg.experiment.nets_dir):
        os.makedirs(cfg.experiment.nets_dir, exist_ok=True)

    if not os.path.exists(cfg.experiment.output_dir):
        os.makedirs(cfg.experiment.output_dir, exist_ok=True)

    cfg.save_yaml(cfg.experiment.output_dir / "config.yaml")

    handle_logging(cfg)
    handle_warnings()

    trainer = Trainer(cfg)

    try:
        trainer.run()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()
