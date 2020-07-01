# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from thtrainer.callbacks import Callback

KEY_SEG = '_'

class TensorBoard(Callback):

    def __init__(self, log_dir=None, comment='', writer=None, step_logs_keys=None, input_to_model=None, **kwargs):
        """
        log_dir (string): Save directory location. Default is
                  runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run.
                  Use hierarchical folder structure to compare
                  between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
                  for each new experiment to compare across them.
        comment (string): Comment log_dir suffix appended to the default
                  ``log_dir``. If ``log_dir`` is assigned, this argument has no effect.
        writer (SummaryWriter)
        step_logs_keys (str, tuple, list): Display metric in tensorboard by step_logs_key.
        input_to_model (Tensor): Display model structure
                  if ``None``, Don't display, else display model structure
        """

        if log_dir is None:
            log_dir = './runs'
        self.input_to_model = input_to_model
        now_date = str(datetime.now())
        for c in ['/', '\\', ':']:
            now_date = now_date.replace(c, KEY_SEG)
        log_dir = os.path.join(log_dir, now_date)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = writer or SummaryWriter(log_dir, comment, **kwargs)

        if isinstance(step_logs_keys, str):
            step_logs_keys = [step_logs_keys]
        elif step_logs_keys is None:
            step_logs_keys = []
        if not isinstance(step_logs_keys, (tuple, list)):
            raise RuntimeError('step_logs_keys only support str or tuple or list, but get', step_logs_keys)
        self.step_logs_keys = step_logs_keys
        self.step_count = 1

    def on_train_begin(self, logs=None):
        if self.input_to_model is not None:
            self.writer.add_graph(
                model=self.model.model,
                input_to_model=self.input_to_model
            )

    def on_batch_end(self, batch, logs=None):
        for k in self.step_logs_keys:
            if k in logs:
                self.writer.add_scalar('step/' + k, logs[k], self.step_count)
                self.step_count += 1

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







