# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import absolute_import

class MetricList(object):
    '''
    # Example
        >>> from thtrainer.metrics import Accuracy, TopKCategoricalAccuracy, Metric, Loss
        >>> import torch
        >>> from torch import nn
        >>> pred = torch.randn(100, 10)
        >>> y = torch.randint(0, 10, (100,))
        >>>
        >>> metrics = MetricList([
        >>>     Accuracy(),
        >>>     TopKCategoricalAccuracy(),
        >>>     Loss(nn.CrossEntropyLoss())
        >>> ])
        >>>
        >>> metrics.update((pred, y))
        >>> print(metrics.compute())
    '''

    def __init__(self, metrics):
        if isinstance(metrics, list):
            _metrics = {}
            for metric in metrics:
                name = metric.__class__.__name__
                _metrics[name] = metric
            metrics = _metrics
        self.metrics = metrics
        self.reset()

    def keys(self):
        return self.metrics.keys()

    def values(self):
        return self.metrics.values()

    def __getitem__(self, item):
        return self.metrics[item]

    def reset(self):
        """
        Resets the metric to it's initial state.

        This is called at the start of each epoch.
        """
        for metric in self.metrics.values():
            metric.reset()

    def started(self):
        for metric in self.metrics.values():
            metric.reset()

    def update(self, output):
        """
        Updates the metric's state using the passed batch output.

        This is called once for each batch.

        Args:
            output: the is the output from the engine's process function.
        """
        for metric in self.metrics.values():
            metric.update(output)

    def compute(self):
        """
        Computes the metric based on it's accumulated state.

        This is called at the end of each epoch.

        Returns:
            Any: the actual quantity of interest.

        Raises:
            NotComputableError: raised when the metric cannot be computed.
        """
        res = {}
        for name, metric in self.metrics.items():
            res[name] = metric.compute()
        return res

    def iteration_completed(self, output):
        for metric in self.metrics.values():
            metric.iteration_completed(output)

    def completed(self, name):
        for metric in self.metrics.values():
            metric.completed(name)







