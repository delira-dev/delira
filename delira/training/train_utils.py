import numpy as np
from ..utils.decorators import dtype_func

from delira import get_backends

if "TORCH" in get_backends():
    import torch

    from ..utils.decorators import torch_tensor_func, torch_module_func

    @dtype_func(float)
    def float_to_pytorch_tensor(f: float):
        """
        Converts a single float to a PyTorch Float-Tensor

        Parameters
        ----------
        f : float
            float to convert

        Returns
        -------
        torch.Tensor
            converted float

        """
        return torch.from_numpy(np.array([f], dtype=np.float32))

    @torch_module_func
    def create_optims_default_pytorch(model, optim_cls, **optim_params):
        """
        Function to create a optimizer dictionary
        (in this case only one optimizer for the whole network)

        Parameters
        ----------
        model : :class:`AbstractPyTorchNetwork`
            model whose parameters should be updated by the optimizer
        optim_cls :
            Class implementing an optimization algorithm
        **optim_params :
            Additional keyword arguments (passed to the optimizer class

        Returns
        -------
        dict
            dictionary containing all created optimizers
        """
        return {"default": optim_cls(model.parameters(), **optim_params)}

    @torch_module_func
    def create_optims_gan_default_pytorch(model, optim_cls, **optim_params):
        """
        Function to create a optimizer dictionary
        (in this case two optimizers: One for the generator,
        one for the discriminator)

        Parameters
        ----------
        model : :class:`AbstractPyTorchNetwork`
            model whose parameters should be updated by the optimizer
        optim_cls :
            Class implementing an optimization algorithm
        optim_params :
            Additional keyword arguments (passed to the optimizer class

        Returns
        -------
        dict
            dictionary containing all created optimizers
        """
        return {"gen": optim_cls(model.gen.parameters(), **optim_params),
                "discr": optim_cls(model.discr.parameters(), **optim_params)}

    def convert_torch_tensor_to_npy(*args, **kwargs):
        """
        Function to convert all torch Tensors to numpy arrays

        Parameters
        ----------
        *args :
            arbitrary positional arguments
        **kwargs :
            arbitrary keyword arguments

        Returns
        -------
        Iterable
            all given positional arguments (converted if necessary)

        dict
            all given keyword arguments (converted if necessary)

        """
        args = [_arg.detach().cpu().numpy() for _arg in args
                if isinstance(_arg, torch.Tensor)]
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.detach().cpu().numpy()

        return args, kwargs

if "TF" in get_backends():
    import tensorflow as tf

    def tf_convert_to_numpy(*args):
        return tuple(_x for _x in args)

    def create_optims_default_tf(optim_cls, **optim_params):
        """
        Function to create a optimizer dictionary
        (in this case only one optimizer)

        Parameters
        ----------
        optim_cls :
            Class implementing an optimization algorithm
        **optim_params :
            Additional keyword arguments (passed to the optimizer class)

        Returns
        -------
        dict
            dictionary containing all created optimizers
        """
        return {"default": optim_cls(**optim_params)}

    def initialize_uninitialized(sess):
        """
        Function to initialize only uninitialized variables in a session graph

        Parameters
        ----------

        sess : tf.Session()
        """

        global_vars = tf.global_variables()
        is_not_initialized = sess.run(
            [tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(
            global_vars, is_not_initialized) if not f]

        if not_initialized_vars:
            sess.run(tf.variables_initializer(not_initialized_vars))

    def convert_tf_tensor_to_npy(*args, **kwargs):
        """
        Function to convert all torch Tensors to numpy arrays

        Parameters
        ----------
        *args :
            arbitrary positional arguments
        **kwargs :
            arbitrary keyword arguments

        Returns
        -------
        Iterable
            all given positional arguments (converted if necessary)

        dict
            all given keyword arguments (converted if necessary)

        """
        return args, kwargs
