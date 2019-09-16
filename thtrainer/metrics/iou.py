# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import absolute_import

from thtrainer.metrics import Metric, ConfusionMatrix, IoU
import torch

class IOUMetric(Metric):
    '''
    # Arguments
        num_classes (int): number of classes. In case of images, num_classes should also count the background index 0.
        average (str, optional): confusion matrix values averaging schema: None, "samples", "recall", "precision".
            Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
            samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
            represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
            diagonal values represent class precisions.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        ignore_index (int, optional): index to ignore, e.g. background index
    '''

    def __init__(self, num_classes,
                 average=None,
                 ignore_index=None,
                 output_transform=lambda x: x):
        self.cm = ConfusionMatrix(num_classes=num_classes, average=average)
        self.iou = IoU(self.cm, ignore_index=ignore_index)
        self.output_transform = output_transform

    def reset(self):
        self.cm.reset()
        self.iou.reset()

    def update(self, output):
        output = self.output_transform(output)
        self.cm.update(output)

    def compute(self):
        res = self.iou.compute()
        if torch.is_tensor(res):
            res = res.cpu().detach().numpy()
        return res


class MeanIOUMetric(IOUMetric):

    def __init__(self, num_classes,
                 average=None,
                 ignore_index=None,
                 output_transform=lambda x: x):
        super(MeanIOUMetric, self).__init__(num_classes,
                                            average,
                                            ignore_index,
                                            output_transform=output_transform)

    def compute(self):
        return super(MeanIOUMetric, self).compute().mean()







