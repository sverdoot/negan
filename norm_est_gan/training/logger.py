"""
Implementation of the Logger object for performing training logging and visualisation.
"""
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_mimicry.training import Logger
from torchvision import utils as vutils


class CustomLogger(Logger):
    def __init__(
        self,
        log_dir,
        num_steps,
        dataset_size,
        device,
        flush_secs=120,
        **kwargs,
    ):
        super().__init__(log_dir, num_steps, dataset_size, device, flush_secs, **kwargs)
        self.writer = self._build_writer()
        self.writers = None

    def _build_writer(self):
        writer = SummaryWriter(
            log_dir=os.path.join(
                self.log_dir,
            ),
            flush_secs=self.flush_secs,
        )

        return writer

    def write_summaries(self, log_data, global_step):
        """
        Tasks appropriate writers to write the summaries in tensorboard. Creates a
        dditional writers for summary writing if there are new scalars to log in
        log_data.

        Args:
            log_data (MetricLog): Dict-like object to collect log data for TB writing.
            global_step (int): Global step variable for syncing logs.

        Returns:
            None
        """
        for metric, data in log_data.items():
            #     if metric not in self.writers:
            #         self.writers[metric] = self._build_writer(metric)

            # Write with a group name if it exists
            # name = log_data.get_group_name(metric) or metric
            self.writer.add_scalar(
                metric,
                log_data[metric],
                global_step=global_step,  # name,
            )

    def close_writers(self):
        """
        Closes all writers.
        """
        # for metric in self.writers:
        #     self.writers[metric].close()
        self.writer.close()

    # def print_log(self, global_step, log_data, time_taken):
    #     """
    #     Formats the string to print to stdout based on training information.

    #     Args:
    #         log_data (MetricLog): Dict-like object to collect log data for TB writing.
    #         global_step (int): Global step variable for syncing logs.
    #         time_taken (float): Time taken for one training iteration.

    #     Returns:
    #         str: String to be printed to stdout.
    #     """
    #     # Basic information
    #     log_to_show = [
    #         "INFO: [Epoch {:d}/{:d}][Global Step: {:d}/{:d}]".format(
    #             self._get_epoch(global_step), self.num_epochs, global_step,
    #             self.num_steps)
    #     ]

    #     # Display GAN information as fed from user.
    #     GAN_info = [""]
    #     metrics = sorted(log_data.keys())

    #     for metric in metrics:
    #         GAN_info.append('{}: {}'.format(metric, log_data[metric]))

    #     # Add train step time information
    #     GAN_info.append("({:.4f} sec/idx)".format(time_taken))

    #     # Accumulate to log
    #     log_to_show.append("\n| ".join(GAN_info))

    #     # Finally print the output
    #     ret = " ".join(log_to_show)
    #     print(ret)

    #     return ret

    # def _get_fixed_noise(self, nz, num_images, output_dir=None):
    #     """
    #     Produce the fixed gaussian noise vectors used across all models
    #     for consistency.
    #     """
    #     if output_dir is None:
    #         output_dir = os.path.join(self.log_dir, 'viz')

    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     output_file = os.path.join(output_dir,
    #                                'fixed_noise_nz_{}.pth'.format(nz))

    #     if os.path.exists(output_file):
    #         noise = torch.load(output_file)

    #     else:
    #         noise = torch.randn((num_images, nz))
    #         torch.save(noise, output_file)

    #     return noise.to(self.device)

    # def _get_fixed_labels(self, num_images, num_classes):
    #     """
    #     Produces fixed class labels for generating fixed images.
    #     """
    #     labels = np.array([i % num_classes for i in range(num_images)])
    #     labels = torch.from_numpy(labels).to(self.device)

    #     return labels

    def vis_images(self, netG, global_step, num_images=64):
        """
        Produce visualisations of the G(z), one fixed and one random.

        Args:
            netG (Module): Generator model object for producing images.
            global_step (int): Global step variable for syncing logs.
            num_images (int): The number of images to visualise.

        Returns:
            None
        """
        img_dir = os.path.join(self.log_dir, "images")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        with torch.no_grad():
            # Generate random images
            noise = torch.randn((num_images, netG.nz), device=self.device)
            fake_images = netG(noise).detach().cpu()

            # Generate fixed random images
            fixed_noise = self._get_fixed_noise(nz=netG.nz, num_images=num_images)

            if hasattr(netG, "num_classes") and netG.num_classes > 0:
                fixed_labels = self._get_fixed_labels(num_images, netG.num_classes)
                fixed_fake_images = netG(fixed_noise, fixed_labels).detach().cpu()
            else:
                fixed_fake_images = netG(fixed_noise).detach().cpu()

            # Map name to results
            images_dict = {"fixed_fake": fixed_fake_images, "fake": fake_images}

            # Visualise all results
            for name, images in images_dict.items():
                images_viz = vutils.make_grid(images, padding=2, normalize=True)

                vutils.save_image(
                    images_viz,
                    f"{img_dir}/{name}_samples_step_{global_step}.png",
                    normalize=True,
                )

                # if 'img' not in self.writers:
                #     self.writers['img'] = self._build_writer('img')

                self.writer.add_image(
                    f"{name}_vis",
                    images_viz,
                    global_step=global_step,
                )
