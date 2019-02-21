import logging
logger = logging.getLogger(__name__)

from delira import get_backends

if "TORCH" in get_backends():
    import torch
    from torchvision import models as t_models

    from delira.models.abstract_network import AbstractPyTorchNetwork


    class GenerativeAdversarialNetworkBasePyTorch(AbstractPyTorchNetwork):
        """Implementation of Vanilla DC-GAN to create 64x64 pixel images

        Notes
        -----
        The fully connected part in the discriminator has been replaced with an
        equivalent convolutional part

        References
        ----------
        https://arxiv.org/abs/1511.06434

        See Also
        --------
        :class:`AbstractPyTorchNetwork`

        """

        def __init__(self, n_channels, noise_length, **kwargs):
            """

            Parameters
            ----------
            n_channels : int
                number of image channels for generated images and input images
            noise_length : int
                length of noise vector
            **kwargs :
                additional keyword arguments

            """

            # register params by passing them as kwargs to parent class __init__
            super().__init__(n_channels=n_channels,
                            noise_length=noise_length,
                            **kwargs)

            gen, discr = self._build_models(n_channels, noise_length, **kwargs)

            self.nz = noise_length

            self.gen = gen
            self.discr = discr

            for key, value in kwargs.items():
                setattr(self, key, value)

        def forward(self, real_image_batch):
            """
            Create fake images by feeding noise through generator and feed results
            and real images through discriminator

            Parameters
            ----------
            real_image_batch : torch.Tensor
                batch of real images

            Returns
            -------
            torch.Tensor
                Generated fake images
            torch.Tensor
                Discriminator prediction of fake images
            torch.Tensor
                Discriminator prediction of real images

            """
            noise = torch.randn(real_image_batch.size(0), self.nz, 1, 1,
                                device=real_image_batch.device)

            fake_image_batch = self.gen(noise)

            discr_pred_fake = self.discr(fake_image_batch)
            discr_pred_real = self.discr(real_image_batch)

            return fake_image_batch, discr_pred_fake, discr_pred_real

        @staticmethod
        def closure(model, data_dict: dict,
                    optimizers: dict, criterions={}, metrics={},
                    fold=0, **kwargs):
            """
            closure method to do a single backpropagation step

            Parameters
            ----------
            model : :class:`ClassificationNetworkBase`
                trainable model
            data_dict : dict
                dictionary containing data
            optimizers : dict
                dictionary of optimizers to optimize model's parameters
            criterions : dict
                dict holding the criterions to calculate errors
                (gradients from different criterions will be accumulated)
            metrics : dict
                dict holding the metrics to calculate
            fold : int
                Current Fold in Crossvalidation (default: 0)
            kwargs : dict
                additional keyword arguments

            Returns
            -------
            dict
                Metric values (with same keys as input dict metrics)
            dict
                Loss values (with same keys as input dict criterions)
            list
                Arbitrary number of predictions as torch.Tensor

            Raises
            ------
            AssertionError
                if optimizers or criterions are empty or the optimizers are not
                specified

            """

            loss_vals = {}
            metric_vals = {}
            total_loss_discr_real = 0
            total_loss_discr_fake = 0
            total_loss_gen = 0

            # choose suitable context manager:
            if optimizers:
                context_man = torch.enable_grad

            else:
                context_man = torch.no_grad

            with context_man():
                batch = data_dict.pop("data")

                # predict batch
                fake_image_batch, discr_pred_fake, discr_pred_real = model(batch)

                # train discr with prediction from real image
                for key, crit_fn in criterions.items():
                    _loss_val = crit_fn(discr_pred_real,
                                        torch.ones_like(discr_pred_real))
                    loss_vals[key + "_discr_real"] = _loss_val.detach()
                    total_loss_discr_real += _loss_val

                # train discr with prediction from fake image
                for key, crit_fn in criterions.items():
                    _loss_val = crit_fn(discr_pred_fake,
                                        torch.zeros_like(discr_pred_fake))
                    loss_vals[key + "_discr_fake"] = _loss_val.detach()
                    total_loss_discr_fake += _loss_val

                total_loss_discr = total_loss_discr_fake + total_loss_discr_real

                if optimizers:

                    # actual backpropagation
                    optimizers["discr"].zero_grad()
                    # perform loss scaling via apex if half precision is enabled
                    with optimizers["discr"].scale_loss(
                            total_loss_discr) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                    optimizers["discr"].step()

                # calculate adversarial loss for generator update
                for key, crit_fn in criterions.items():
                    _loss_val = crit_fn(discr_pred_fake,
                                        torch.ones_like(discr_pred_fake))
                    loss_vals[key + "_adversarial"] = _loss_val.detach().cpu()
                    total_loss_gen += _loss_val

                with torch.no_grad():
                    for key, metric_fn in metrics.items():
                        # calculate metrics for discriminator with real prediction
                        metric_vals[key + "_discr_real"] = metric_fn(
                            discr_pred_real,
                            torch.ones_like(
                                discr_pred_real)).detach()

                        # calculate metrics for discriminator with fake prediction
                        metric_vals[key + "_discr_fake"] = metric_fn(
                            discr_pred_fake,
                            torch.zeros_like(
                                discr_pred_fake)).detach()

                        # calculate adversarial metrics
                        metric_vals[key + "_adversarial"] = metric_fn(
                            discr_pred_fake,
                            torch.ones_like(
                                discr_pred_fake)).detach()

                if optimizers:
                    # actual backpropagation
                    optimizers["gen"].zero_grad()
                    # perform loss scaling via apex if half precision is enabled
                    with optimizers["gen"].scale_loss(
                            total_loss_gen) as scaled_loss:
                        scaled_loss.backward()
                    optimizers["gen"].step()

                else:
                    # add prefix "val" in validation mode
                    eval_loss_vals, eval_metrics_vals = {}, {}
                    for key in loss_vals.keys():
                        eval_loss_vals["val_" + str(key)] = loss_vals[key]

                    for key in metric_vals:
                        eval_metrics_vals["val_" + str(key)] = metric_vals[key]

                    loss_vals = eval_loss_vals
                    metric_vals = eval_metrics_vals

            for key, val in {**metric_vals, **loss_vals}.items():
                logging.info({"value": {"value": val.item(), "name": key,
                                        "env_appendix": "_%02d" % fold
                                        }})

            logging.info({'image_grid': {"images": batch, "name": "real_images",
                                        "env_appendix": "_%02d" % fold}})
            logging.info({"image_grid": {"images": fake_image_batch,
                                        "name": "fake_images",
                                        "env_appendix": "_%02d" % fold}})

            return metric_vals, loss_vals, [fake_image_batch, discr_pred_fake,
                                            discr_pred_real]

        @staticmethod
        def _build_models(in_channels, noise_length, **kwargs):
            """
            Builds actual generator and discriminator models

            Parameters
            ----------
            in_channels : int
                number of channels for generated images by generator and inputs of
                discriminator
            noise_length : int
                length of noise vector (generator input)
            **kwargs :
                additional keyword arguments

            Returns
            -------
            torch.nn.Sequential
                generator
            torch.nn.Sequential
                discriminator
            """
            gen = torch.nn.Sequential(
                    # input is Z, going into a convolution
                    torch.nn.ConvTranspose2d(noise_length, 64 * 8, 4, 1, 0, bias=False),
                    torch.nn.BatchNorm2d(64 * 8),
                    torch.nn.ReLU(True),
                    # state size. (64*8) x 4 x 4
                    torch.nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
                    torch.nn.BatchNorm2d(64 * 4),
                    torch.nn.ReLU(True),
                    # state size. (64*4) x 8 x 8
                    torch.nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
                    torch.nn.BatchNorm2d(64 * 2),
                    torch.nn.ReLU(True),
                    # state size. (64*2) x 16 x 16
                    torch.nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(True),
                    # state size. (64) x 32 x 32
                    torch.nn.ConvTranspose2d(64, in_channels, 4, 2, 1, bias=False),
                    torch.nn.Tanh()
                    # state size. (nc) x 64 x 64
                )

            discr = torch.nn.Sequential(
                        # input is (nc) x 64 x 64
                        torch.nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
                        torch.nn.LeakyReLU(0.2, inplace=True),
                        # state size. (64) x 32 x 32
                        torch.nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                        torch.nn.BatchNorm2d(64 * 2),
                        torch.nn.LeakyReLU(0.2, inplace=True),
                        # state size. (64*2) x 16 x 16
                        torch.nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                        torch.nn.BatchNorm2d(64 * 4),
                        torch.nn.LeakyReLU(0.2, inplace=True),
                        # state size. (64*4) x 8 x 8
                        torch.nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
                        torch.nn.BatchNorm2d(64 * 8),
                        torch.nn.LeakyReLU(0.2, inplace=True),
                        # state size. (64*8) x 4 x 4
                        torch.nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
                        torch.nn.Sigmoid()
                    )

            return gen, discr
