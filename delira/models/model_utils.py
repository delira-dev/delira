import contextlib

try:
    # use apex loss scaling if possible
    # (and enabled, this is done internally by apex)
    from apex import amp
except ImportError:
    # use no loss scaling with same API if apex is unavailable
    amp = None


@contextlib.contextmanager
def scale_loss(loss,
               optimizers,
               loss_id=0,
               model=None,
               delay_unscale=False,
               **kwargs):
    if amp is None:
        yield loss

    else:
        with amp.scale_loss(loss=loss, optimizers=optimizers,
                            loss_id=loss_id, model=model,
                            delay_unscale=delay_unscale, 
                            **kwargs) as _loss:
            yield _loss
    return
