# Adapted from https://github.com/jaxony/unet-pytorch/blob/master/model.py

from delira import get_backends

if "TORCH" in get_backends():
    import torch
    import torch.nn.functional as F
    from torch.nn import init
    import logging
    from ..abstract_network import AbstractPyTorchNetwork


    class UNet2dPyTorch(AbstractPyTorchNetwork):
        """
        The :class:`UNet2dPyTorch` is a convolutional encoder-decoder neural
        network.
        Contextual spatial information (from the decoding,
        expansive pathway) about an input tensor is merged with
        information representing the localization of details
        (from the encoding, compressive pathway).

        Notes
        -----
        Differences to the original paper:

            * padding is used in 3x3 convolutions to prevent loss of border pixels
            * merging outputs does not require cropping due to (1)
            * residual connections can be used by specifying ``merge_mode='add'``
            * if non-parametric upsampling is used in the decoder pathway (
                specified by upmode='upsample'), then an additional 1x1 2d
                convolution occurs after upsampling to reduce channel
                dimensionality by a factor of 2. This channel halving happens
                with the convolution in the tranpose convolution (specified by
                ``upmode='transpose'``)

        References
        ----------
        https://arxiv.org/abs/1505.04597

        See Also
        --------
        :class:`UNet3dPyTorch`

        """

        def __init__(self, num_classes, in_channels=1, depth=5,
                    start_filts=64, up_mode='transpose',
                    merge_mode='concat'):
            """

            Parameters
            ----------
            num_classes : int
                number of output classes
            in_channels : int
                number of channels for the input tensor (default: 1)
            depth : int
                number of MaxPools in the U-Net (default: 5)
            start_filts : int
                number of convolutional filters for the first conv (affects all
                other conv-filter numbers too; default: 64)
            up_mode : str
                type of upconvolution. Must be one of ['transpose', 'upsample']
                if 'transpose':
                    Use transpose convolution for upsampling
                if 'upsample':
                    Use bilinear Interpolation for upsampling (no additional
                    trainable parameters)
                default: 'transpose'
            merge_mode : str
                mode of merging the two paths (with and without pooling). Must
                be one of ['merge', 'add']
                if 'merge':
                    Concatenates along the channel dimension (Original UNet)
                if 'add':
                    Adds both tensors (Residual behaviour)
                default: 'merge'

            """

            super().__init__()

            if up_mode in ('transpose', 'upsample'):
                self.up_mode = up_mode
            else:
                raise ValueError("\"{}\" is not a valid mode for "
                                "upsampling. Only \"transpose\" and "
                                "\"upsample\" are allowed.".format(up_mode))

            if merge_mode in ('concat', 'add'):
                self.merge_mode = merge_mode
            else:
                raise ValueError("\"{}\" is not a valid mode for"
                                "merging up and down paths. "
                                "Only \"concat\" and "
                                "\"add\" are allowed.".format(up_mode))

            # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
            if self.up_mode == 'upsample' and self.merge_mode == 'add':
                raise ValueError("up_mode \"upsample\" is incompatible "
                                "with merge_mode \"add\" at the moment "
                                "because it doesn't make sense to use "
                                "nearest neighbour to reduce "
                                "depth channels (by half).")

            self.num_classes = num_classes
            self.in_channels = in_channels
            self.start_filts = start_filts
            self.depth = depth

            self.down_convs = []
            self.up_convs = []

            self.conv_final = None

            self._build_model(num_classes, in_channels, depth, start_filts)

            self.reset_params()

        @staticmethod
        def weight_init(m):
            """
            Initializes weights with xavier_normal and bias with zeros

            Parameters
            ----------
            m : torch.nn.Module
                module to initialize

            """
            if isinstance(m, torch.nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

        def reset_params(self):
            """
            Initialize all parameters

            """
            for i, m in enumerate(self.modules()):
                self.weight_init(m)

        def forward(self, x):
            """
            Feed tensor through network

            Parameters
            ----------
            x : torch.Tensor

            Returns
            -------
            torch.Tensor
                Prediction

            """
            encoder_outs = []

            # encoder pathway, save outputs for merging
            for i, module in enumerate(self.down_convs):
                x, before_pool = module(x)
                encoder_outs.append(before_pool)

            for i, module in enumerate(self.up_convs):
                before_pool = encoder_outs[-(i + 2)]
                x = module(before_pool, x)

            # No softmax is used. This means you need to use
            # torch.nn.CrossEntropyLoss is your training script,
            # as this module includes a softmax already.
            x = self.conv_final(x)
            return x

        @staticmethod
        def closure(model, data_dict: dict, optimizers: dict, criterions={},
                    metrics={}, fold=0, **kwargs):
            """
            closure method to do a single backpropagation step


            Parameters
            ----------
            model : :class:`ClassificationNetworkBasePyTorch`
                trainable model
            data_dict : dict
                dictionary containing the data
            optimizers : dict
                dictionary of optimizers to optimize model's parameters
            criterions : dict
                dict holding the criterions to calculate errors
                (gradients from different criterions will be accumulated)
            metrics : dict
                dict holding the metrics to calculate
            fold : int
                Current Fold in Crossvalidation (default: 0)
            **kwargs:
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

            assert (optimizers and criterions) or not optimizers, \
                "Criterion dict cannot be emtpy, if optimizers are passed"

            loss_vals = {}
            metric_vals = {}
            total_loss = 0

            # choose suitable context manager:
            if optimizers:
                context_man = torch.enable_grad

            else:
                context_man = torch.no_grad

            with context_man():

                inputs = data_dict.pop("data")
                preds = model(inputs)

                if data_dict:

                    for key, crit_fn in criterions.items():
                        _loss_val = crit_fn(preds, *data_dict.values())
                        loss_vals[key] = _loss_val.detach()
                        total_loss += _loss_val

                    with torch.no_grad():
                        for key, metric_fn in metrics.items():
                            metric_vals[key] = metric_fn(
                                preds, *data_dict.values())

            if optimizers:
                optimizers['default'].zero_grad()
                # perform loss scaling via apex if half precision is enabled
                with optimizers["default"].scale_loss(total_loss) as scaled_loss:
                    scaled_loss.backward()
                optimizers['default'].step()

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

            logging.info({'image_grid': {"images": inputs, "name": "input_images",
                                        "env_appendix": "_%02d" % fold}})

            logging.info({'image_grid': {"images": preds,
                                        "name": "predicted_images",
                                        "env_appendix": "_%02d" % fold}})

            return metric_vals, loss_vals, [preds]

        def _build_model(self, num_classes, in_channels=3, depth=5,
                        start_filts=64):
            """
            Builds the actual model

            Parameters
            ----------
            num_classes : int
                number of output classes
            in_channels : int
                number of channels for the input tensor (default: 1)
            depth : int
                number of MaxPools in the U-Net (default: 5)
            start_filts : int
                number of convolutional filters for the first conv (affects all
                other conv-filter numbers too; default: 64)

            Notes
            -----
            The Helper functions and classes are defined within this function
            because ``delira`` offers a possibility to save the source code
            along the weights to completely recover the network without needing
            a manually created network instance and these helper functions have
            to be saved too.

            """

            def conv3x3(in_channels, out_channels, stride=1,
                        padding=1, bias=True, groups=1):
                return torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=groups)

            def upconv2x2(in_channels, out_channels, mode='transpose'):
                if mode == 'transpose':
                    return torch.nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=2,
                        stride=2)
                else:
                    # out_channels is always going to be the same
                    # as in_channels
                    return torch.nn.Sequential(
                        torch.nn.Upsample(mode='bilinear', scale_factor=2),
                        conv1x1(in_channels, out_channels))

            def conv1x1(in_channels, out_channels, groups=1):
                return torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    groups=groups,
                    stride=1)

            class DownConv(torch.nn.Module):
                """
                A helper Module that performs 2 convolutions and 1 MaxPool.
                A ReLU activation follows each convolution.
                """

                def __init__(self, in_channels, out_channels, pooling=True):
                    super(DownConv, self).__init__()

                    self.in_channels = in_channels
                    self.out_channels = out_channels
                    self.pooling = pooling

                    self.conv1 = conv3x3(self.in_channels, self.out_channels)
                    self.conv2 = conv3x3(self.out_channels, self.out_channels)

                    if self.pooling:
                        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    before_pool = x
                    if self.pooling:
                        x = self.pool(x)
                    return x, before_pool

            class UpConv(torch.nn.Module):
                """
                A helper Module that performs 2 convolutions and 1 UpConvolution.
                A ReLU activation follows each convolution.
                """

                def __init__(self, in_channels, out_channels,
                            merge_mode='concat', up_mode='transpose'):
                    super(UpConv, self).__init__()

                    self.in_channels = in_channels
                    self.out_channels = out_channels
                    self.merge_mode = merge_mode
                    self.up_mode = up_mode

                    self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                            mode=self.up_mode)

                    if self.merge_mode == 'concat':
                        self.conv1 = conv3x3(
                            2 * self.out_channels, self.out_channels)
                    else:
                        # num of input channels to conv2 is same
                        self.conv1 = conv3x3(self.out_channels,
                                            self.out_channels)
                    self.conv2 = conv3x3(self.out_channels, self.out_channels)

                def forward(self, from_down, from_up):
                    from_up = self.upconv(from_up)
                    if self.merge_mode == 'concat':
                        x = torch.cat((from_up, from_down), 1)
                    else:
                        x = from_up + from_down
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    return x

            outs = in_channels
            # create the encoder pathway and add to a list
            for i in range(depth):
                ins = self.in_channels if i == 0 else outs
                outs = start_filts * (2 ** i)
                pooling = True if i < depth - 1 else False

                down_conv = DownConv(ins, outs, pooling=pooling)
                self.down_convs.append(down_conv)

            # create the decoder pathway and add to a list
            # - careful! decoding only requires depth-1 blocks
            for i in range(depth - 1):
                ins = outs
                outs = ins // 2
                up_conv = UpConv(ins, outs, up_mode=self.up_mode,
                                merge_mode=self.merge_mode)
                self.up_convs.append(up_conv)

            self.conv_final = conv1x1(outs, num_classes)

            # add the list of modules to current module
            self.down_convs = torch.nn.ModuleList(self.down_convs)
            self.up_convs = torch.nn.ModuleList(self.up_convs)

        @staticmethod
        def prepare_batch(batch: dict, input_device, output_device):
            """
            Helper Function to prepare Network Inputs and Labels (convert them to
            correct type and shape and push them to correct devices)

            Parameters
            ----------
            batch : dict
                dictionary containing all the data
            input_device : torch.device
                device for network inputs
            output_device : torch.device
                device for network outputs

            Returns
            -------
            dict
                dictionary containing data in correct type and shape and on correct
                device

            """
            return_dict = {"data": torch.from_numpy(batch.pop("data")).to(
                input_device).to(torch.float)}

            for key, vals in batch.items():
                if key == "label" and len(vals.shape) == 4:
                    vals = vals[:, 0]  # remove first axis if to many axis
                                    # (channel dimension)
                return_dict[key] = torch.from_numpy(vals).to(output_device).to(
                    torch.long)

            return return_dict


    class UNet3dPyTorch(AbstractPyTorchNetwork):
        """
        The :class:`UNet3dPyTorch` is a convolutional encoder-decoder neural
        network.
        Contextual spatial information (from the decoding,
        expansive pathway) about an input tensor is merged with
        information representing the localization of details
        (from the encoding, compressive pathway).

        Notes
        -----
        Differences to the original paper:
            * Working on 3D data instead of 2D slices
            * padding is used in 3x3x3 convolutions to prevent loss of border
                pixels
            * merging outputs does not require cropping due to (1)
            * residual connections can be used by specifying ``merge_mode='add'``
            * if non-parametric upsampling is used in the decoder pathway (
                specified by upmode='upsample'), then an additional 1x1x1 3d
                convolution occurs after upsampling to reduce channel
                dimensionality by a factor of 2. This channel halving happens
                with the convolution in the tranpose convolution (specified by
                ``upmode='transpose'``)

        References
        ----------
        https://arxiv.org/abs/1505.04597

        See Also
        --------
        :class:`UNet2dPyTorch`

        """

        def __init__(self, num_classes, in_channels=3, depth=5,
                    start_filts=64, up_mode='transpose',
                    merge_mode='concat'):
            """

            Parameters
            ----------
            num_classes : int
                number of output classes
            in_channels : int
                number of channels for the input tensor (default: 1)
            depth : int
                number of MaxPools in the U-Net (default: 5)
            start_filts : int
                number of convolutional filters for the first conv (affects all
                other conv-filter numbers too; default: 64)
            up_mode : str
                type of upconvolution. Must be one of ['transpose', 'upsample']
                if 'transpose':
                    Use transpose convolution for upsampling
                if 'upsample':
                    Use trilinear Interpolation for upsampling (no additional
                    trainable parameters)
                default: 'transpose'
            merge_mode : str
                mode of merging the two paths (with and without pooling). Must
                be one of ['merge', 'add']
                if 'merge':
                    Concatenates along the channel dimension (Original UNet)
                if 'add':
                    Adds both tensors (Residual behaviour)
                default: 'merge'

            """
            super().__init__()

            if up_mode in ('transpose', 'upsample'):
                self.up_mode = up_mode
            else:
                raise ValueError("\"{}\" is not a valid mode for "
                                "upsampling. Only \"transpose\" and "
                                "\"upsample\" are allowed.".format(up_mode))

            if merge_mode in ('concat', 'add'):
                self.merge_mode = merge_mode
            else:
                raise ValueError("\"{}\" is not a valid mode for"
                                "merging up and down paths. "
                                "Only \"concat\" and "
                                "\"add\" are allowed.".format(up_mode))

            # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
            if self.up_mode == 'upsample' and self.merge_mode == 'add':
                raise ValueError("up_mode \"upsample\" is incompatible "
                                "with merge_mode \"add\" at the moment "
                                "because it doesn't make sense to use "
                                "nearest neighbour to reduce "
                                "depth channels (by half).")

            self.num_classes = num_classes
            self.in_channels = in_channels
            self.start_filts = start_filts
            self.depth = depth

            self.down_convs = []
            self.up_convs = []
            self.conv_final = None

            self._build_model(num_classes, in_channels, depth, start_filts)

            self.reset_params()

        @staticmethod
        def weight_init(m):
            """
            Initializes weights with xavier_normal and bias with zeros

            Parameters
            ----------
            m : torch.nn.Module
                module to initialize

            """

            if isinstance(m, torch.nn.Conv3d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

        def reset_params(self):
            """
            Initialize all parameters

            """
            for i, m in enumerate(self.modules()):
                self.weight_init(m)

        def forward(self, x):
            """
            Feed tensor through network

            Parameters
            ----------
            x : torch.Tensor

            Returns
            -------
            torch.Tensor
                Prediction

            """
            encoder_outs = []

            # encoder pathway, save outputs for merging
            for i, module in enumerate(self.down_convs):
                x, before_pool = module(x)
                encoder_outs.append(before_pool)

            for i, module in enumerate(self.up_convs):
                before_pool = encoder_outs[-(i + 2)]
                x = module(before_pool, x)

            # No softmax is used. This means you need to use
            # torch.nn.CrossEntropyLoss is your training script,
            # as this module includes a softmax already.
            x = self.conv_final(x)
            return x

        @staticmethod
        def closure(model, data_dict: dict, optimizers: dict, criterions={},
                    metrics={}, fold=0, **kwargs):
            """
            closure method to do a single backpropagation step


            Parameters
            ----------
            model : :class:`ClassificationNetworkBasePyTorch`
                trainable model
            data_dict : dict
                dictionary containing the data
            optimizers : dict
                dictionary of optimizers to optimize model's parameters
            criterions : dict
                dict holding the criterions to calculate errors
                (gradients from different criterions will be accumulated)
            metrics : dict
                dict holding the metrics to calculate
            fold : int
                Current Fold in Crossvalidation (default: 0)
            **kwargs:
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

            assert (optimizers and criterions) or not optimizers, \
                "Criterion dict cannot be emtpy, if optimizers are passed"

            loss_vals = {}
            metric_vals = {}
            total_loss = 0

            # choose suitable context manager:
            if optimizers:
                context_man = torch.enable_grad

            else:
                context_man = torch.no_grad

            with context_man():

                inputs = data_dict.pop("data")
                preds = model(inputs)

                if data_dict:

                    for key, crit_fn in criterions.items():
                        _loss_val = crit_fn(preds, *data_dict.values())
                        loss_vals[key] = _loss_val.detach()
                        total_loss += _loss_val

                    with torch.no_grad():
                        for key, metric_fn in metrics.items():
                            metric_vals[key] = metric_fn(
                                preds, *data_dict.values())

            if optimizers:
                optimizers['default'].zero_grad()
                # perform loss scaling via apex if half precision is enabled
                with optimizers["default"].scale_loss(total_loss) as scaled_loss:
                    scaled_loss.backward()
                optimizers['default'].step()

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

            slicing_dim = inputs.size(2) // 2  # visualize slice in mid of volume

            logging.info({'image_grid': {"inputs": inputs[:, :, slicing_dim, ...],
                                        "name":
                                            "input_images",
                                        "env_appendix": "_%02d" % fold}})

            logging.info({'image_grid': {"results": preds[:, :, slicing_dim, ...],
                                        "name":
                                            "predicted_images",
                                        "env_appendix": "_%02d" % fold}})

            return metric_vals, loss_vals, [preds]

        def _build_model(self, num_classes, in_channels=3, depth=5,
                        start_filts=64):
            """
            Builds the actual model

            Parameters
            ----------
            num_classes : int
                number of output classes
            in_channels : int
                number of channels for the input tensor (default: 1)
            depth : int
                number of MaxPools in the U-Net (default: 5)
            start_filts : int
                number of convolutional filters for the first conv (affects all
                other conv-filter numbers too; default: 64)

            Notes
            -----
            The Helper functions and classes are defined within this function
            because ``delira`` offers a possibility to save the source code
            along the weights to completely recover the network without needing
            a manually created network instance and these helper functions have
            to be saved too.

            """

            def conv3x3x3(in_channels, out_channels, stride=1,
                        padding=1, bias=True, groups=1):
                return torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=groups)

            def upconv2x2x2(in_channels, out_channels, mode='transpose'):
                if mode == 'transpose':
                    return torch.nn.ConvTranspose3d(
                        in_channels,
                        out_channels,
                        kernel_size=2,
                        stride=2)
                else:
                    # out_channels is always going to be the same
                    # as in_channels
                    return torch.nn.Sequential(
                        torch.nn.Upsample(mode='trilinear', scale_factor=2),
                        conv1x1x1(in_channels, out_channels))

            def conv1x1x1(in_channels, out_channels, groups=1):
                return torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    groups=groups,
                    stride=1)

            class DownConv(torch.nn.Module):
                """
                A helper Module that performs 2 convolutions and 1 MaxPool.
                A ReLU activation follows each convolution.
                """

                def __init__(self, in_channels, out_channels, pooling=True):
                    super(DownConv, self).__init__()

                    self.in_channels = in_channels
                    self.out_channels = out_channels
                    self.pooling = pooling

                    self.conv1 = conv3x3x3(self.in_channels, self.out_channels)
                    self.conv2 = conv3x3x3(self.out_channels, self.out_channels)

                    if self.pooling:
                        self.pool = torch.nn.MaxPool3d(kernel_size=2, stride=2)

                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    before_pool = x
                    if self.pooling:
                        x = self.pool(x)
                    return x, before_pool

            class UpConv(torch.nn.Module):
                """
                A helper Module that performs 2 convolutions and 1 UpConvolution.
                A ReLU activation follows each convolution.
                """

                def __init__(self, in_channels, out_channels,
                            merge_mode='concat', up_mode='transpose'):
                    super(UpConv, self).__init__()

                    self.in_channels = in_channels
                    self.out_channels = out_channels
                    self.merge_mode = merge_mode
                    self.up_mode = up_mode

                    self.upconv = upconv2x2x2(self.in_channels, self.out_channels,
                                            mode=self.up_mode)

                    if self.merge_mode == 'concat':
                        self.conv1 = conv3x3x3(
                            2 * self.out_channels, self.out_channels)
                    else:
                        # num of input channels to conv2 is same
                        self.conv1 = conv3x3x3(self.out_channels,
                                            self.out_channels)
                    self.conv2 = conv3x3x3(self.out_channels, self.out_channels)

                def forward(self, from_down, from_up):
                    from_up = self.upconv(from_up)
                    if self.merge_mode == 'concat':
                        x = torch.cat((from_up, from_down), 1)
                    else:
                        x = from_up + from_down
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    return x

            outs = in_channels
            # create the encoder pathway and add to a list
            for i in range(depth):
                ins = self.in_channels if i == 0 else outs
                outs = start_filts * (2 ** i)
                pooling = True if i < depth - 1 else False

                down_conv = DownConv(ins, outs, pooling=pooling)
                self.down_convs.append(down_conv)

            # create the decoder pathway and add to a list
            # - careful! decoding only requires depth-1 blocks
            for i in range(depth - 1):
                ins = outs
                outs = ins // 2
                up_conv = UpConv(ins, outs, up_mode=self.up_mode,
                                merge_mode=self.merge_mode)
                self.up_convs.append(up_conv)

            self.conv_final = conv1x1x1(outs, num_classes)

            # add the list of modules to current module
            self.down_convs = torch.nn.ModuleList(self.down_convs)
            self.up_convs = torch.nn.ModuleList(self.up_convs)

        @staticmethod
        def prepare_batch(batch: dict, input_device, output_device):
            """
            Helper Function to prepare Network Inputs and Labels (convert them to
            correct type and shape and push them to correct devices)

            Parameters
            ----------
            batch : dict
                dictionary containing all the data
            input_device : torch.device
                device for network inputs
            output_device : torch.device
                device for network outputs

            Returns
            -------
            dict
                dictionary containing data in correct type and shape and on correct
                device

            """
            return_dict = {"data": torch.from_numpy(batch.pop("data")).to(
                input_device).to(torch.float)}

            for key, vals in batch.items():
                if key == "label" and len(vals.shape) == 5:
                    vals = vals[:, 0]  # remove first axis if to many axis
                    # (channel dimension)
                return_dict[key] = torch.from_numpy(vals).to(output_device).to(
                    torch.long)

            return return_dict
