'''
Test Trainer

Test Callbacks:
TerminateOnNaN, ProgbarLogger,
ModelCheckpoint, EarlyStopping,
CSVLogger, Trainer

'''

# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import absolute_import

from unittest import TestCase

from torch import nn
import torch as th
from torch.utils.data import Dataset

from nn_torch.detection.utils.trainer import (
    TerminateOnNaN, ProgbarLogger,
    ModelCheckpoint, EarlyStopping,
    CSVLogger, Trainer
)


class TestModel(nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()
        self.linear = nn.Linear(10, 1)
        self.test_nan = 0
        self.test_TerminateOnNaN = False


    def forward(self, x):
        if self.test_nan > 5 and self.test_TerminateOnNaN:
            return self.linear(x) / 0
        self.test_nan += 1
        return self.linear(x)

class DS(Dataset):

    def __len__(self):
        return 20

    def __getitem__(self, idx):
        return th.randn(10), th.randn(1)


class TestTrainer(TestCase):

    def setUp(self) -> None:
        self.model = TestModel()
        self.optim = th.optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        self.ds = DS()


    def test_TerminateOnNaN(self):
        trainer = Trainer(self.model, self.optim,
                          self.loss_fn,
                          [TerminateOnNaN()],
                          )
        self.model.test_TerminateOnNaN = True

        trainer.fit(self.ds, epochs=2, verbose=1)


    def test_ProgbarLogger(self):
        trainer = Trainer(self.model, self.optim,
                          self.loss_fn,
                          # [ProgbarLogger()]
                          )

        trainer.fit(self.ds, epochs=2, verbose=1)


    def test_ModelCheckpoint(self):
        trainer = Trainer(self.model, self.optim,
                          self.loss_fn,
                          [ModelCheckpoint(
                              './tmp/test_mdoel.th',
                              monitor='loss'
                          )],)

        trainer.fit(self.ds, epochs=2)


    def test_EarlyStopping(self):
        trainer = Trainer(self.model, self.optim,
                          self.loss_fn,
                          [EarlyStopping(monitor='loss', verbose=1)],
                          )

        trainer.fit(self.ds, epochs=10, verbose=1)

    def test_CSVLogger(self):
        trainer = Trainer(self.model, self.optim,
                          self.loss_fn,
                          [CSVLogger('./tmp/logs.csv')],
                          )

        trainer.fit(self.ds, epochs=10, verbose=1)



