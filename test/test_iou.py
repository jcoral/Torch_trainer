# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import absolute_import
from unittest import TestCase

from thtrainer.metrics import *
import torch

class TestIOU(TestCase):

    def test_miou(self):
        cm = ConfusionMatrix(2)
        metric = mIoU(cm)
        pred = torch.randn(2, 2, 300, 300)
        target = torch.randint(0, 2, (2, 300, 300))
        cm.update((pred, target))
        print(metric.compute())


    def test_iou_metric(self):
        metric = IOUMetric(2)
        pred = torch.randn(2, 2, 300, 300)
        target = torch.randint(0, 2, (2, 300, 300))
        metric.update((pred, target))
        print(metric.compute())

    def test_miou_metric(self):
        metric = MeanIOUMetric(2)

        for i in range(10):
            pred = torch.randn(2, 2, 300, 300)
            target = torch.randint(0, 2, (2, 300, 300))
            metric.update((pred, target))
        print(metric.compute())









