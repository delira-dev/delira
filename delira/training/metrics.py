from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from ..utils.decorators import make_deprecated

import trixi

from delira import get_backends

if "TORCH" in get_backends():
    import torch

    from .train_utils import pytorch_tensor_to_numpy, float_to_pytorch_tensor 
    @make_deprecated(trixi)
    class AurocMetricPyTorch(torch.nn.Module):
        """
        Metric to Calculate AuROC

        .. deprecated:: 0.1
            :class:`AurocMetricPyTorch` will be removed in next release and is
            deprecated in favor of ``trixi.logging`` Modules

        .. warning::
            :class:`AurocMetricPyTorch` will be removed in next release

        """
        def __init__(self):
            super().__init__()

        def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
            """
            Actual AuROC calculation

            Parameters
            ----------
            outputs : torch.Tensor
                predictions from network
            targets : torch.Tensor
                training targets

            Returns
            -------
            torch.Tensor
                auroc value

            """
            if outputs.dim() == 2:
                outputs = torch.argmax(outputs, dim=1)
            score = roc_auc_score(pytorch_tensor_to_numpy(targets),
                                pytorch_tensor_to_numpy(outputs))
            return float_to_pytorch_tensor(score)


    @make_deprecated(trixi)
    class AccuracyMetricPyTorch(torch.nn.Module):
        """
        Metric to Calculate Accuracy
        
        .. deprecated:: 0.1
            :class:`AccuracyMetricPyTorch` will be removed in next release and is
            deprecated in favor of ``trixi.logging`` Modules

        .. warning::
            class:`AccuracyMetricPyTorch` will be removed in next release

        """
        def __init__(self, normalize=True, sample_weight=None):
            """

            Parameters
            ----------
            normalize : bool, optional (default=True)
            If ``False``, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

            sample_weight : array-like of shape = [n_samples], optional
                Sample weights.

            """
            super().__init__()
            self.normalize = normalize
            self.sample_weight = sample_weight

        def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
            """
            Actual accuracy calcuation

            Parameters
            ----------
            outputs : torch.Tensor
                predictions from network
            targets : torch.Tensor
                training targets

            Returns
            -------
            torch.Tensor
                accuracy value

            """
            outputs = outputs > 0.5
            if outputs.dim() == 2:
                outputs = torch.argmax(outputs, dim=1)
            score = accuracy_score(pytorch_tensor_to_numpy(targets),
                                pytorch_tensor_to_numpy(outputs),
                                self.normalize, self.sample_weight)
            return float_to_pytorch_tensor(score)
