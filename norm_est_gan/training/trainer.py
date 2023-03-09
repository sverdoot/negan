import os
import re
import time
from typing import Sequence

import torch
from torch_mimicry.training import Trainer, metric_log

from .callback import Callback
from .logger import CustomLogger


class CustomTrainer(Trainer):
    def __init__(
        self,
        netD,
        netG,
        optD,
        optG,
        dataloader,
        num_steps,
        log_dir="./log",
        n_dis=1,
        lr_decay=None,
        device=None,
        netG_ckpt_file=None,
        netD_ckpt_file=None,
        print_steps=1,
        vis_steps=500,
        log_steps=50,
        save_steps=5000,
        flush_secs=30,
        callbacks: Sequence[Callback] = [],
    ):
        super().__init__(
            netD,
            netG,
            optD,
            optG,
            dataloader,
            num_steps,
            log_dir,
            n_dis,
            lr_decay,
            device,
            netG_ckpt_file,
            netD_ckpt_file,
            print_steps,
            vis_steps,
            log_steps,
            save_steps,
            flush_secs,
        )
        self.callbacks = callbacks
        self.logger = CustomLogger(
            log_dir=self.log_dir,
            num_steps=self.num_steps,
            dataset_size=len(self.dataloader),
            flush_secs=flush_secs,
            device=self.device,
        )

    def train(self):
        """
        Runs the training pipeline with all given parameters in Trainer.
        """
        # Restore models
        global_step = self._restore_models_and_step()
        print("INFO: Starting training from global step {}...".format(global_step))
        for callback in self.callbacks:
            callback.cnt = global_step

        try:
            start_time = time.time()

            # Iterate through data
            iter_dataloader = iter(self.dataloader)
            while global_step < self.num_steps:
                log_data = metric_log.MetricLog()  # log data for tensorboard

                # -------------------------
                #   One Training Step
                # -------------------------
                # Update n_dis times for D
                for i in range(self.n_dis):
                    iter_dataloader, real_batch = self._fetch_data(
                        iter_dataloader=iter_dataloader
                    )

                    # ------------------------
                    #   Update D Network
                    # -----------------------
                    log_data = self.netD.train_step(
                        real_batch=real_batch,
                        netG=self.netG,
                        optD=self.optD,
                        log_data=log_data,
                        global_step=global_step,
                        device=self.device,
                    )

                    # -----------------------
                    #   Update G Network
                    # -----------------------
                    # Update G, but only once.
                    if i == (self.n_dis - 1):
                        log_data = self.netG.train_step(
                            real_batch=real_batch,
                            netD=self.netD,
                            optG=self.optG,
                            global_step=global_step,
                            log_data=log_data,
                            device=self.device,
                        )

                # --------------------------------
                #   Update Training Variables
                # -------------------------------
                global_step += 1

                log_data = self.scheduler.step(
                    log_data=log_data, global_step=global_step
                )

                # -------------------------
                #   Logging and Metrics
                # -------------------------
                for callback in self.callbacks:
                    callback.invoke({"step": global_step}, log_data)

                if global_step % self.log_steps == 0:
                    self.logger.write_summaries(
                        log_data=log_data, global_step=global_step
                    )

                if global_step % self.print_steps == 0:
                    curr_time = time.time()
                    self.logger.print_log(
                        global_step=global_step,
                        log_data=log_data,
                        time_taken=(curr_time - start_time) / self.print_steps,
                    )
                    start_time = curr_time

                if global_step % self.vis_steps == 0:
                    self.logger.vis_images(netG=self.netG, global_step=global_step)

                if global_step % self.save_steps == 0:
                    print("INFO: Saving checkpoints...")
                    self._save_model_checkpoints(global_step)

            print("INFO: Saving final checkpoints...")
            self._save_model_checkpoints(global_step)

        except KeyboardInterrupt:
            print("INFO: Saving checkpoints from keyboard interrupt...")
            self._save_model_checkpoints(global_step)

        finally:
            self.logger.close_writers()

        print("INFO: Training Ended.")
