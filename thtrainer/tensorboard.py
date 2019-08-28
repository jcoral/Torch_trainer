# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import absolute_import

from thtrainer.callbacks import Callback
from torch.utils.tensorboard import SummaryWriter
import torch

class TensorBoard(Callback):

    def __init__(self, log_dir=None, comment='', writer=None, input_to_model=None, **kwargs):
        self.input_to_model = input_to_model
        self.writer = writer or SummaryWriter(log_dir, comment, **kwargs)

    def on_train_begin(self, logs=None):
        if self.input_to_model is not None:
            self.writer.add_graph(
                model=self.model.model,
                input_to_model=self.input_to_model
            )

    def on_epoch_end(self, epoch, logs=None):
        # Split logs
        if logs is None or len(logs) == 0:
            return
        metrics_logs = {metric.split(':')[-1]: {} for metric in self.params['metrics']}

        for k, v in logs.items():
            is_metric_key = False
            seg_k = k.split(':')
            for metric_key in metrics_logs:
                if metric_key == seg_k[-1]:
                    is_metric_key = True
                    metrics_logs[metric_key][seg_k[0]] = v
            if not is_metric_key:
                self.writer.add_scalar(k, v, epoch)

        for metric, values in metrics_logs.items():
            self.writer.add_scalars(metric, values, epoch)

        # self.writer.add_scalars('Train', logs, epoch)

    def on_train_end(self, logs=None):
        self.writer.close()






