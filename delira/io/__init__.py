try:

    from .torch import save_checkpoint as torch_save_checkpoint
    from .torch import load_checkpoint as torch_load_checkpoint

    __torch_io = [
        'torch_save_checkpoint'
        'torch_load_checkpoint'
    ]

except ModuleNotFoundError as e:
    import warnings
    warnings.warn(e)
    raise e

    __torch_io = []

__all__ = [
    *__torch_io
]