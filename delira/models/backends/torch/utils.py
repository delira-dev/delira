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
    """
    Context Manager which automatically switches between loss scaling via
    apex.amp (if apex is available) and no loss scaling

    Parameters
    ----------
    loss : :class:`torch.Tensor`
        a pytorch tensor containing the loss value
    optimizers : list
        a list of :class:`torch.optim.Optimizer` containing all optimizers,
        which are holding paraneters affected by the backpropagation of the
        current loss value
    loss_id : int
        When used in conjunction with the ``num_losses`` argument
        to ``amp.initialize``, enables Amp to use a different loss scale per
        loss.  ``loss_id`` must be an integer between 0 and ``num_losses`` that
        tells Amp which loss is being used for the current backward pass.
        If ``loss_id`` is left unspecified, Amp will use the default global
        loss scaler for this backward pass.
    model : :class:`AbstractPyTorchNetwork` or None
        Currently unused, reserved to enable future optimizations.
    delay_unscale : bool
        ``delay_unscale`` is never necessary, and the default value of
        ``False`` is strongly recommended. If ``True``, Amp will not unscale
        the gradients or perform model->master gradient copies on
        context manager exit. ``delay_unscale=True`` is a minor ninja
        performance optimization and can result
        in weird gotchas (especially with multiple models/optimizers/losses),
        so only use it if you know what you're doing.
    **kwargs :
        additional keyword arguments; currently unused, but provided for the
        case amp decides to extend the functionality here

    Yields
    ------
    :class:`torch.Tensor`
        the new loss value (scaled if apex.amp is available and was configured
        to do so, unscaled in all other cases)

    """

    if amp is None:
        yield loss

    else:
        with amp.scale_loss(loss=loss, optimizers=optimizers,
                            loss_id=loss_id, model=model,
                            delay_unscale=delay_unscale,
                            **kwargs) as _loss:

            yield _loss
