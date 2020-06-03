# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from thtrainer.callbacks import Callback

KEY_SEG = '_'

class TensorBoard(Callback):

    def __init__(self, log_dir=None, comment='', writer=None, input_to_model=None, **kwargs):
        if log_dir is None:
            log_dir = './'
        self.input_to_model = input_to_model
        now_date = str(datetime.now())
        for c in ['/', '\\', ':']:
            now_date = now_date.replace(c, KEY_SEG)
        log_dir = os.path.join(log_dir, now_date)
        print(log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
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

        metrics_logs = {}
        for metric in logs.keys():
            metric_segs = metric.split(':')
            name = metric_segs[0]
            if len(metric_segs) > 1:
                name = KEY_SEG.join(metric_segs[1:])
            metrics_logs[name] = {}

        for k, v in logs.items():
            is_metric_key = False
            seg_k = k.split(':')
            if len(seg_k) > 1:
                seg_k = [seg_k[0], KEY_SEG.join(seg_k[1:])]

            for metric_key in metrics_logs:
                if metric_key == seg_k[-1]:
                    is_metric_key = True
                    metric_key = metric_key.replace(':', KEY_SEG)
                    tag = seg_k[0].replace(':', KEY_SEG)
                    metrics_logs[metric_key][tag] = v
            if not is_metric_key:
                metric_key = k.replace(':', KEY_SEG)
                self.writer.add_scalar(metric_key, v, epoch)

        for metric, values in metrics_logs.items():
            self.writer.add_scalars(metric, values, epoch)

    def on_train_end(self, logs=None):
        self.writer.close()







