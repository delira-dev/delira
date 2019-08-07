from delira import get_backends

if "TORCH" in get_backends():
    from delira.io.torch import save_checkpoint_torch as torch_save_checkpoint
    from delira.io.torch import load_checkpoint_torch as torch_load_checkpoint

    from delira.io.torch import save_checkpoint_torchscript \
        as torchscript_save_checkpoint
    from delira.io.torch import load_checkpoint_torchscript \
        as torchscript_load_checkpoint

if "TF" in get_backends():
    from delira.io.tf import save_checkpoint as tf_save_checkpoint
    from delira.io.tf import load_checkpoint as tf_load_checkpoint

    from delira.io.tf import save_checkpoint_eager as tf_eager_save_checkpoint
    from delira.io.tf import load_checkpoint_eager as tf_eager_load_checkpoint

if "CHAINER" in get_backends():
    from delira.io.chainer import save_checkpoint as chainer_save_checkpoint
    from delira.io.chainer import load_checkpoint as chainer_load_checkpoint

if "SKLEARN" in get_backends():
    from delira.io.sklearn import load_checkpoint as sklearn_load_checkpoint
    from delira.io.sklearn import save_checkpoint as sklearn_save_checkpoint
