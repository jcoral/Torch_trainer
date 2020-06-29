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
import torch
from torch.utils.data import Dataset
import time

from thtrainer.callbacks import (
    TerminateOnNaN, ProgbarLogger,
    ModelCheckpoint, EarlyStopping,
    CSVLogger, Callback
)

from thtrainer.tensorboard import TensorBoard

from thtrainer.metrics import *

from thtrainer.trainer import Trainer


class CustomCallback(Callback):

    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            logs['cus_key'] = int(batch + 1)


class TestModel(nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 10)
        self.test_nan = 0
        self.test_TerminateOnNaN = False


    def forward(self, x):
        if self.test_nan > 5 and self.test_TerminateOnNaN:
            return self.linear(x) / 0
        self.test_nan += 1
        if self.test_nan % 3 == 0:
            time.sleep(1)

        for linear in [self.linear1, self.linear2, self.linear3]:
            x = linear(x)
        return x

class DS(Dataset):

    def __len__(self):
        return 20

    def __getitem__(self, idx):
        return torch.randn(10), torch.randint(0, 10, (1,))[0]


class TestTrainer(TestCase):

    def setUp(self) -> None:
        self.model = TestModel()
        self.optim = torch.optim.Adam(self.model.parameters())
        self.loss_fn = nn.CrossEntropyLoss()
        self.ds = DS()


    def test_TerminateOnNaN(self):
        trainer = Trainer(
            self.model, self.optim,
            self.loss_fn,
            [TerminateOnNaN()],
        )
        self.model.test_TerminateOnNaN = True

        trainer.fit(self.ds, epochs=1, verbose=1)


    def test_ProgbarLogger(self):
        trainer = Trainer(
            self.model, self.optim,
            self.loss_fn,
            metrics=[
                Accuracy(),
                TopKCategoricalAccuracy(),
                'Loss'
            ],
            val_metrics=[Accuracy()],
            callbacks=[CustomCallback()]
        )

        history = trainer.fit(self.ds, batch_size=2,
                              epochs=10, verbose=1,
                              validation_data=self.ds, logs=['cus_key'])
        print(history.history)


    def test_ModelCheckpoint(self):
        trainer = Trainer(self.model, self.optim,
                          self.loss_fn,
                          [ModelCheckpoint(
                              '../tmp/test_model.pth',
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
                          [CSVLogger('../tmp/logs.csv')],
                          )

        history = trainer.fit(self.ds, epochs=10, verbose=1)
        print(history.history)


    def test_tensorboard(self):
        ipt = torch.stack([self.ds[i][0] for i in range(2)])
        trainer = Trainer(
            self.model, self.optim,
            self.loss_fn,
            callbacks=[TensorBoard('/Volumes/Coral/tmp/nb_tmp',
                                           comment='Test',
                                           input_to_model=ipt)],
            metrics=[
                Accuracy(),
                TopKCategoricalAccuracy(),
                'loss'
            ],
            val_metrics=[
                Accuracy(),
                TopKCategoricalAccuracy(),
                'loss'
            ]
        )

        history = trainer.fit(
            self.ds, batch_size=2,
            epochs=10, verbose=1,
            validation_data=self.ds
        )



