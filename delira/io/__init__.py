from delira import get_backends

if "TORCH" in get_backends():
    from .torch import save_checkpoint_torch as torch_save_checkpoint
    from .torch import load_checkpoint_torch as torch_load_checkpoint
    from .torch import save_checkpoint_torchscript \
        as torchscript_save_checkpoint
    from .torch import load_checkpoint_torchscript \
        as torchscript_load_checkpoint

if "TF" in get_backends():
    from .tf import save_checkpoint as tf_save_checkpoint
    from .tf import load_checkpoint as tf_load_checkpoint

<<<<<<< HEAD
    from .tf import save_checkpoint_eager as tf_eager_save_checkpoint
    from .tf import load_checkpoint_eager as tf_eager_load_checkpoint

=======
>>>>>>> 0ea4e6354ded64add4137cb202eb0a34645f75b0
if "CHAINER" in get_backends():
    from .chainer import save_checkpoint as chainer_save_checkpoint
    from .chainer import load_checkpoint as chainer_load_checkpoint
