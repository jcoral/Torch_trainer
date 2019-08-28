# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import absolute_import

from pprint import pprint
from unittest import TestCase

import torch
from torch.utils.data import DataLoader

from thtrainer.metrics import coco


def prepare():
    test_transform = make_transforms(train=False)
    dataset = dataset # dataset from COCOMetricDataset
    data_loader = DataLoader(dataset, 16, shuffle=False, collate_fn=coco.COCOMetric.collate_fn)
    metric = coco.COCOMetric(data_loader,
                             img_size=(300, 300),
                             output_transform=coco.bbox_transform)
    model = DetctionNetwork()

    model.eval()
    return model, data_loader, metric


class TestCOCOMetric(TestCase):

    def test_reset(self):
        print('-' * 43, '  test_reset  ', '-' * 43)
        model, data_loader, metric = prepare()

        with torch.no_grad():
            for imgs, target in data_loader:
                pred = model(imgs)
                metric.update((pred, target))
                break

        res = metric.compute()

        print('-' * 20, '  reset  ', '-' * 20)
        metric.reset()
        i = 0
        with torch.no_grad():
            for imgs, target in data_loader:
                i += 1
                if i == 1: continue
                pred = model(imgs)
                metric.update((pred, target))
                break
        res = metric.compute()

        print('-'* 100)

    def test_update(self):
        print('-' * 43, '  test_update  ', '-' * 43)
        model, data_loader, metric = prepare()

        with torch.no_grad():
            for imgs, target in data_loader:
                pred = model(imgs)
                metric.update((pred, target))
                break

        res = metric.compute()
        print('-'* 100)


    def test_compute(self):
        print('-' * 43, '  test_reset  ', '-' * 43)
        model, data_loader, metric = prepare()

        i = 0
        with torch.no_grad():
            for imgs, target in data_loader:
                pred = model(imgs)
                metric.update((pred, target))
                i += 1
                if i == 2:
                    break
        res = metric.compute()
        print('-'* 100)







