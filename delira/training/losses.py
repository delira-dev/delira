from delira import get_backends

if "TORCH" in get_backends():
    import torch
    import torch.nn.functional as F

    class BCEFocalLossPyTorch(torch.nn.Module):
        """
        Focal loss for binary case without(!) logit

        """

        def __init__(self, alpha=None, gamma=2, reduction='elementwise_mean'):
            """
            Implements Focal Loss for binary class case

            Parameters
            ----------
            alpha : float
                alpha has to be in range [0,1], assigns class weight
            gamma : float
                focusing parameter
            reduction : str
                Specifies the reduction to apply to the output: ‘none’ |
                ‘elementwise_mean’ | ‘sum’. ‘none’: no reduction will be applied,
                ‘elementwise_mean’: the sum of the output will be divided by the
                number of elements in the output, ‘sum’: the output will be summed
            (further information about parameters above can be found in pytorch
            documentation)

            Returns
            -------
            torch.Tensor
                loss value

            """
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, p, t):
            bce_loss = F.binary_cross_entropy(p, t, reduction='none')

            if self.alpha is not None:
                # create weights for alpha
                alpha_weight = torch.ones(t.shape, device=p.device) * \
                    self.alpha
                alpha_weight = torch.where(torch.eq(t, 1.),
                                           alpha_weight, 1 - alpha_weight)
            else:
                alpha_weight = torch.Tensor([1]).to(p.device)

            # create weights for focal loss
            focal_weight = 1 - torch.where(torch.eq(t, 1.), p, 1 - p)
            focal_weight.pow_(self.gamma)
            focal_weight.to(p.device)

            # compute loss
            focal_loss = focal_weight * alpha_weight * bce_loss

            if self.reduction == 'elementwise_mean':
                return torch.mean(focal_loss)
            if self.reduction == 'none':
                return focal_loss
            if self.reduction == 'sum':
                return torch.sum(focal_loss)
            raise AttributeError('Reduction parameter unknown.')

    class BCEFocalLossLogitPyTorch(torch.nn.Module):
        """
        Focal loss for binary case WITH logit

        """

        def __init__(self, alpha=None, gamma=2, reduction='elementwise_mean'):
            """
            Implements Focal Loss for binary class case

            Parameters
            ----------
            alpha : float
                alpha has to be in range [0,1], assigns class weight
            gamma : float
                focusing parameter
            reduction : str
                Specifies the reduction to apply to the output: ‘none’ |
                ‘elementwise_mean’ | ‘sum’. ‘none’: no reduction will be applied,
                ‘elementwise_mean’: the sum of the output will be divided by the
                number of elements in the output, ‘sum’: the output will be summed
            (further information about parameters above can be found in pytorch
            documentation)

            Returns
            -------
            torch.Tensor
                loss value

            """
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, p, t):
            bce_loss = F.binary_cross_entropy_with_logits(
                p, t, reduction='none')

            p = torch.sigmoid(p)

            if self.alpha is not None:
                # create weights for alpha
                alpha_weight = torch.ones(t.shape, device=p.device) * \
                    self.alpha
                alpha_weight = torch.where(torch.eq(t, 1.),
                                           alpha_weight, 1 - alpha_weight)
            else:
                alpha_weight = torch.Tensor([1]).to(p.device)

            # create weights for focal loss
            focal_weight = 1 - torch.where(torch.eq(t, 1.), p, 1 - p)
            focal_weight.pow_(self.gamma)
            focal_weight.to(p.device)

            # compute loss
            focal_loss = focal_weight * alpha_weight * bce_loss

            if self.reduction == 'elementwise_mean':
                return torch.mean(focal_loss)
            if self.reduction == 'none':
                return focal_loss
            if self.reduction == 'sum':
                return torch.sum(focal_loss)
            raise AttributeError('Reduction parameter unknown.')
