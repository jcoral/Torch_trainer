# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import absolute_import

from unittest import TestCase

from thtrainer.metrics import Accuracy, TopKCategoricalAccuracy, Metric, Loss, MetricList
import torch
from torch import nn

class TestMetricList(TestCase):
    def test_compute(self):

        pred = torch.randn(100, 10)
        y = torch.randint(0, 10, (100,))

        metrics = MetricList([
            Accuracy(),
            TopKCategoricalAccuracy(),
            Loss(nn.CrossEntropyLoss())
        ])

        metrics.update((pred, y))
        print(metrics.compute())
