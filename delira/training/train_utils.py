import numpy as np
from ..utils.decorators import dtype_func

try:
    import torch

    from ..utils.decorators import torch_tensor_func, torch_module_func

    @torch_tensor_func
    def pytorch_batch_to_numpy(tensor: torch.Tensor):
        """
        Utility Function to cast a whole PyTorch batch to numpy

        Parameters
        ----------
        tensor : torch.Tensor
            the batch to convert

        Returns
        -------
        np.ndarray
            the converted batch

        """
        return np.array([pytorch_tensor_to_numpy(tmp[0]) for tmp in tensor.split(1)])

    @torch_tensor_func
    def pytorch_tensor_to_numpy(tensor: torch.Tensor):
        """
        Utility Function to cast a single PyTorch Tensor to numpy

        Parameters
        ----------
        tensor : torch.Tensor
            the tensor to convert

        Returns
        -------
        np.ndarray
            the converted tensor

        """
        return tensor.detach().cpu().numpy()

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

except ImportError as e:
    raise e
