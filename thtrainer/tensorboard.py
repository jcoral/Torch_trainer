# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import absolute_import

from thtrainer.callbacks import Callback
from torch.utils.tensorboard import SummaryWriter

class TensorBoard(Callback):

    def __init__(self, writer=None, log_dir=None, comment='', **kwargs):
        if writer is None:
            writer = SummaryWriter(log_dir, comment, **kwargs)
        self.writer = writer

    def on_train_begin(self, logs=None):
        self.writer.add_graph(model=self.model.model)

    def on_epoch_end(self, epoch, logs=None):
        self.writer.add_scalars('Train log', logs)






