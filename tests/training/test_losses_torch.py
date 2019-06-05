
import unittest

from delira import get_backends


class FocalLossTestPyTorch(unittest.TestCase):

    @unittest.skipIf("TORCH" not in get_backends(),
                     reason="No torch backend installed")
    def test_focalloss(self):
        """
        Test some predefines focal loss values
        """

        from delira.training.losses import BCEFocalLossLogitPyTorch, \
            BCEFocalLossPyTorch
        import torch.nn as nn
        import torch
        import torch.nn.functional as F

        # examples
        #######################################################################
        # binary values
        p = torch.Tensor([[0, 0.2, 0.5, 1.0], [0, 0.2, 0.5, 1.0]])
        t = torch.Tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
        p_l = torch.Tensor([[-2, -1, 0, 2], [-2, -1, 0, 1]])

        #######################################################################
        # params
        gamma = 2
        alpha = 0.25
        eps = 1e-8

        #######################################################################
        # compute targets
        # target for focal loss
        p_t = p * t + (1 - p) * (1 - t)
        alpha_t = torch.Tensor([alpha]).expand_as(t) * t + \
            (1 - t) * (1 - torch.Tensor([alpha]).expand_as(t))
        w = alpha_t * (1 - p_t).pow(torch.Tensor([gamma]))
        fc_value = F.binary_cross_entropy(p, t, w, reduction='none')

        # target for focal loss with logit
        p_tmp = torch.sigmoid(p_l)
        p_t = p_tmp * t + (1 - p_tmp) * (1 - t)
        alpha_t = torch.Tensor([alpha]).expand_as(t) * t + \
            (1 - t) * (1 - torch.Tensor([alpha]).expand_as(t))
        w = alpha_t * (1 - p_t).pow(torch.Tensor([gamma]))

        fc_value_logit = \
            F.binary_cross_entropy_with_logits(p_l, t, w, reduction='none')

        #######################################################################
        # test against BCE and CE =>focal loss with gamma=0, alpha=None
        # test against binary_cross_entropy
        bce = nn.BCELoss(reduction='none')
        focal = BCEFocalLossPyTorch(alpha=None, gamma=0, reduction='none')
        bce_loss = bce(p, t)
        focal_loss = focal(p, t)

        self.assertTrue((torch.abs(bce_loss - focal_loss) < eps).all())

        # test against binary_cross_entropy with logit
        bce = nn.BCEWithLogitsLoss()
        focal = BCEFocalLossLogitPyTorch(alpha=None, gamma=0)
        bce_loss = bce(p_l, t)
        focal_loss = focal(p_l, t)
        self.assertTrue((torch.abs(bce_loss - focal_loss) < eps).all())

        #######################################################################
        # test focal loss with pre computed values
        # test focal loss binary (values manually pre computed)
        focal = BCEFocalLossPyTorch(gamma=gamma, alpha=alpha, reduction='none')
        focal_loss = focal(p, t)
        self.assertTrue((torch.abs(fc_value - focal_loss) < eps).all())

        # test focal loss binary with logit (values manually pre computed)
        # Note that now p_l is used as prediction
        focal = BCEFocalLossLogitPyTorch(
            gamma=gamma, alpha=alpha, reduction='none')
        focal_loss = focal(p_l, t)
        self.assertTrue((torch.abs(fc_value_logit - focal_loss) < eps).all())

        #######################################################################
        # test if backward function works
        p.requires_grad = True
        focal = BCEFocalLossPyTorch(gamma=gamma, alpha=alpha)
        focal_loss = focal(p, t)
        try:
            focal_loss.backward()
        except BaseException:
            self.assertTrue(False, "Backward function failed for focal loss")

        p_l.requires_grad = True
        focal = BCEFocalLossLogitPyTorch(gamma=gamma, alpha=alpha)
        focal_loss = focal(p_l, t)
        try:
            focal_loss.backward()
        except BaseException:
            self.assertTrue(
                False, "Backward function failed for focal loss with logits")


if __name__ == "__main__":
    unittest.main()
