# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

import os

import numpy as np
import torch as th
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader

from thtrainer.callbacks import ProgbarLogger, CallbackList, History
from thtrainer import metrics as trainer_metrics
from thtrainer.metrics import MetricList

metric_dict = {
    'acc': trainer_metrics.Accuracy,
    'accuracy': trainer_metrics.Accuracy,
    'loss': trainer_metrics.Loss,
    'top-k': trainer_metrics.TopKCategoricalAccuracy,
    'mae': trainer_metrics.MeanAbsoluteError,
    'mse': trainer_metrics.MeanSquaredError
}

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return th.optim.lr_scheduler.LambdaLR(optimizer, f)


def _check_data_loader(data, batch_size, shuffle):
    if not isinstance(data, DataLoader):
        data = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data

def _check_metrics(metrics, loss_fn):
    def get_metric_ins(m):
        new_m = []

        for i, mi in enumerate(m):
            if isinstance(mi, str):
                mi = mi.lower()
                if mi not in metric_dict:
                    raise RuntimeError('Not %s metric' % mi)
                if mi  == 'loss':
                    _metric_ins = metric_dict[mi](loss_fn)
                else:
                    _metric_ins = metric_dict[mi]()
                new_m.append(_metric_ins)
            else:
                new_m.append(mi)
        return new_m

    if metrics is None:
        m = {}
        if loss_fn is not None:
            m['loss'] = metric_dict['loss'](loss_fn)
        metrics = MetricList(m)

    if isinstance(metrics, dict):
        metrics = dict(zip(metrics.keys(),
                           get_metric_ins(metrics.values())))
        metrics = MetricList(metrics)

    if isinstance(metrics, (list, tuple)):
        metrics = get_metric_ins(metrics)
        metrics = MetricList(metrics)

    if not isinstance(metrics, trainer_metrics.MetricList):
        raise RuntimeError('Metrics not support %s type' % str(type(metrics)))
    return metrics


class Trainer:
    '''
    Example::
        >>> data_loader = DataLoader(dataset, 10, True, collate_fn=collate_fn, num_workers=4)
        >>>
        >>> optim = th.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
        >>> scheduler = LRSchedulerCallback(StepLR(optim, step_size=4, gamma=0.5))
        >>>
        >>> trainer = Trainer(model, optim,None, callbacks=[scheduler])
        >>> trainer.train_on_batch = train_on_batch(trainer) # Custom train_on_batch
        >>> trainer.fit(data_loader, epochs=50)

    '''

    def __init__(self, model: Module,
                 optimizer: Optimizer,
                 loss_fn: Module,
                 callbacks=None,
                 metrics=None,
                 device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []
        self.metrics = _check_metrics(metrics, loss_fn)

        if device is None:
            self.device = 'cpu'
            if th.cuda.is_available():
                self.device = 'cuda'
        else:
            self.device = device

        self._stop_training = False


    def fit(self, data_loader,
            epochs=1, batch_size=32,
            verbose=1,
            validation_data=None,
            shuffle=True):
        '''
        :param data_loader: Dataset or Dataloader
        :return: Train logs
        '''
        data_loader = _check_data_loader(data_loader, batch_size, shuffle)

        if verbose:
            if self.callbacks is not None:
                self.callbacks = list(self.callbacks) + [ProgbarLogger()]
        history = History()
        self.callbacks.append(history)

        callbacks = CallbackList(self.callbacks)
        callbacks.set_model(self)

        n_steps = len(data_loader)
        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            'steps': n_steps,
            'samples': n_steps,
            'verbose': verbose,
            'metrics': self.metrics.keys(),
        })
        if validation_data is not None:
            callbacks.set_validation_data(validation_data)

        # ref detection warmup
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        warmup_scheduler = warmup_lr_scheduler(
            self.optimizer,
            warmup_iters,
            warmup_factor
        )

        train_loss = {}
        callbacks.on_train_begin(train_loss)
        for epoch in range(1, epochs+1):
            self.model.train()
            if self._stop_training:
                callbacks.on_train_end({})
                return history

            epoch_logs = {}
            callbacks.on_epoch_begin(epoch, epoch_logs)

            batch_log = {}
            self._train_data_loader(epoch, data_loader, callbacks, batch_log, warmup_scheduler)

            for k, v in batch_log.items():
                epoch_logs[k] = v
            # TODO: Add metrics result to epoch_logs
            # TODO: eval validation_data
            callbacks.on_epoch_end(epoch, epoch_logs)

        callbacks.on_train_end(train_loss)
        return history

    def _train_data_loader(self, epoch, data_loader, callbacks, batch_log, warmup_scheduler=None):
        batch_idx = -1
        for batch_data in zip(data_loader):
            batch_idx += 1
            if self._stop_training: return batch_log

            callbacks.on_batch_begin(batch_idx, batch_log)

            loss = self.train_on_batch(*batch_data[0])
            batch_log['loss'] = loss.item()
            callbacks.on_batch_end(batch_idx, batch_log)
            if epoch == 1 and warmup_scheduler is not None:
                warmup_scheduler.step()

        return batch_log

    def train_on_batch(self, X, y=None):
        X = X.to(self.device)
        if y is not None:
            y = y.to(self.device)

        def closure():
            self.optimizer.zero_grad()
            output = self.model(X)
            self.metrics.update((output, y))
            loss = self.loss_fn(output, y)
            loss.backward()
            return loss
        return self.optimizer.step(closure)

    def predict(self, X):
        self.model.eval()
        return self.model(X)

    def stop_training(self):
        self._stop_training = True

    def save_weights(self, filepath, overwrite=True):
        if not overwrite and os.path.exists(filepath): return
        th.save(self.model.state_dict(), filepath)

    def save(self, filepath, overwrite=True):
        if not overwrite and os.path.exists(filepath): return
        th.save(self.model, filepath)

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)







