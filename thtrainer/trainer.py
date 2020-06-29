# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

import os
from copy import deepcopy

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from thtrainer import metrics as trainer_metrics
from thtrainer.callbacks import ProgbarLogger, CallbackList, History
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

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


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
                if mi == 'loss':
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


def _check_logs_param(logs, loss_fn):
    if logs is None:
        logs = []
    if isinstance(logs, (tuple, list)):
        logs = list(logs)
    else:
        raise RuntimeError('logs param must be tuple or list, but get', logs)

    if loss_fn is not None:
        logs.append('loss')

    return logs


def _check_progbar_logger_metrics(metrics, validation_data, logs):
    keys = metrics.keys()
    keys = list(keys)
    if validation_data is None:
        return keys + logs

    train_keys = ['train:' + k for k in keys]
    val_keys = ['val:' + k for k in keys]

    return train_keys + val_keys + logs


def _check_progbar_logger_iters(key, value):
    kvs = {}
    key_t = '%s:%s'
    if isinstance(value, dict):
        value = value.items()
    elif isinstance(value, (tuple, list)):
        value = enumerate(value)
    else:
        return {key: value}

    for i, vi in value:
        _sub_key = key_t % (key, str(i))
        if isinstance(vi, (tuple, list, dict)):
            res = _check_progbar_logger_iters(_sub_key, vi)
        else:
            res = {_sub_key: vi}
        kvs.update(res)
    return kvs


def _check_progbar_logger_value(key, value):
    kvs = {}
    if key is None or key == '':
        key_t = '%s%s'
    else:
        key_t = '%s:%s'
    if isinstance(value, dict):
        value = value.items()
    elif isinstance(value, (tuple, list)):
        value = enumerate(value)
    else:
        return {key: value}

    for i, vi in value:
        _sub_key = key_t % (key, str(i))
        if isinstance(vi, (tuple, list, dict)):
            res = _check_progbar_logger_iters(_sub_key, vi)
        else:
            res = {_sub_key: vi}
        kvs.update(res)
    return kvs


class Trainer:
    '''
    # Arguments
        model: `Module`.
        optimizer: `Optimizer`.
        loss_fn: `Module`.
            Compute model loss
        callbacks: List of `thtrainer.callbacks.Callback` instances.
            List of callbacks to apply during training.
        metrics: List of `thtrainer.metrics.Metric` instances.
            Compute metrics at the end of each batch.
        val_metrics: List of `thtrainer.metrics.Metric` instances.
            Evaluate validation data use val_metrics.
            if `None`: use metrics


    # Example
        >>> data_loader = DataLoader(dataset, 10, True, collate_fn=collate_fn, num_workers=4)
        >>>
        >>> optim = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
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
                 val_metrics=None,
                 device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []
        self.metrics = _check_metrics(metrics, loss_fn)
        if val_metrics is None:
            m = []
            self.val_metrics = MetricList(m)
        else:
            self.val_metrics = _check_metrics(val_metrics, loss_fn)

        if device is None:
            self.device = 'cpu'
            if torch.cuda.is_available():
                self.device = 'cuda'
        else:
            self.device = device

        self._stop_training = False
        self.validation_data = None

    def fit(self, data_loader,
            epochs=1, batch_size=32,
            verbose=1,
            validation_data=None,
            logs=None,
            shuffle=True,
            validate_steps=1,
            validate_init=1,
            warmup=False):
        '''
        # Arguments
            batch_size: Integer.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
            epochs: Integer.
                Number of epochs to train the model.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            validation_data: `DataLoader` or `Dataset`
                Evaluate validation_data the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
            logs: List or tuple
                Keys added by the user in the logs of Batch or Epoch,
                if you need to be displayed in ProgressBar or TensorBoard,
                you need to make logs equal to Keys.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
            validate_steps: Integer
                Evaluate validation data when epoch % validate_steps == 0
            validate_init: Integer
                Evaluate validation data when epoch == validate_init

        '''
        data_loader = _check_data_loader(data_loader, batch_size, shuffle)
        self.validation_data = validation_data

        if verbose:
            if self.callbacks is not None:
                self.callbacks = list(self.callbacks) + [ProgbarLogger()]
        history = History()
        self.callbacks.append(history)

        callbacks = CallbackList(self.callbacks)
        callbacks.set_model(self)

        logs = _check_logs_param(logs, self.loss_fn)

        n_steps = len(data_loader)
        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            'steps': n_steps,
            'samples': n_steps,
            'verbose': verbose,
            'metrics': _check_progbar_logger_metrics(self.metrics, validation_data, logs),
        })
        if validation_data is not None:
            callbacks.set_validation_data(validation_data)

        # ref detection warmup
        warmup_scheduler = None
        if warmup:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(data_loader) - 1)
            warmup_scheduler = warmup_lr_scheduler(
                self.optimizer,
                warmup_iters,
                warmup_factor
            )

        train_logs = {}
        callbacks.on_train_begin(train_logs)
        for epoch in range(1, epochs+1):
            self.model.train()
            if self._stop_training:
                callbacks.on_train_end({})
                return history

            epoch_logs = {}
            callbacks.on_epoch_begin(epoch, epoch_logs)
            self._train_data_loader(epoch, data_loader, callbacks, epoch_logs, warmup_scheduler)

            # evaluate validattion_data
            eval_val_data = epoch % validate_steps == 0 and epoch >= validate_init
            if validation_data is not None and eval_val_data:
                if verbose > 0:
                    print('\nStarting evaluate model')
                eval_res = self.evaluate(
                    validation_data,
                    batch_size,
                    shuffle=False,
                    metrics=self.val_metrics,
                    verbose=0,
                    device=self.device,
                    logs=logs
                )
                res = _check_progbar_logger_value('val', eval_res)
                epoch_logs.update(res)

            callbacks.on_epoch_end(epoch, epoch_logs)

        callbacks.on_train_end(train_logs)
        return history

    def _train_data_loader(self, epoch, data_loader, callbacks, epoch_logs, warmup_scheduler=None):
        self.metrics.reset()
        prefix = ''
        if self.validation_data is not None:
            prefix = 'train'

        loss_sum = 0
        batch_idx = -1
        batch_log = {}
        for batch_data in zip(data_loader):
            batch_idx += 1
            if self._stop_training:
                return batch_log

            callbacks.on_batch_begin(batch_idx, batch_log)

            loss = self.train_on_batch(*batch_data[0])

            batch_log['loss'] = loss.item()
            loss_sum += loss.item()
            callbacks.on_batch_end(batch_idx, batch_log)

            if epoch == 1 and warmup_scheduler is not None:
                warmup_scheduler.step()
        res = self.metrics.compute()
        res = _check_progbar_logger_value(prefix, res)
        epoch_logs.update(res)
        return batch_log

    def train_on_batch(self, X, y=None):
        X = X.to(self.device)
        if y is not None:
            y = y.to(self.device)

        def closure():
            self.optimizer.zero_grad()
            output = self.model(X)
            if y is None:
                pair = (output, )
            else:
                pair = (output, y)
            self.metrics.update(pair)
            loss = self.loss_fn(*pair)
            loss.backward()
            return loss
        return self.optimizer.step(closure)

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        return self.model(X)

    def evaluate_batch(self, X, y=None, device=None):
        device = device or self.device
        X = X.to(device)
        if y is not None:
            y = y.to(device)
        output = self.model(X)
        loss = None
        if self.loss_fn is not None:
            if y is None:
                loss = self.loss_fn(output)
            else:
                loss = self.loss_fn(output, y)
        if y is None:
            if loss is None:
                return output
            else:
                return loss, output
        else:
            if loss is None:
                return (output, y)
            else:
                return loss, (output, y)

    @torch.no_grad()
    def evaluate(self,
                 data_loader, batch_size=1,
                 shuffle=False, metrics=None,
                 verbose=1, device=None,
                 logs=None):
        '''
        # Arguments
            data_loader: `DataLoader`
            batch_size: `Integer` DataLoader batch size
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
            metrics: List of `thtrainer.metrics.Metric` instances.
                Compute metrics at the end of each batch.
            verbose: `Integer`. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            logs: List or tuple
                Keys added by the user in the logs of Batch or Epoch,
                if you need to be displayed in ProgressBar or TensorBoard,
                you need to make logs equal to Keys.
        '''

        data_loader = _check_data_loader(data_loader, batch_size, shuffle)
        if not isinstance(metrics, trainer_metrics.MetricList):
            metrics = _check_metrics(metrics, self.loss_fn)
        n_steps = len(data_loader)

        logs = _check_logs_param(logs, self.loss_fn)

        if verbose > 0:
            progress = ProgbarLogger(stateful_metrics=metrics.keys())
            progress.set_params({
                'batch_size': batch_size,
                'epochs': 1,
                'steps': n_steps,
                'samples': n_steps,
                'verbose': verbose,
                'metrics': list(set(['eval:' + k for k in self.metrics.keys()] + ['loss'])) + logs,
            })

        device = device or self.device
        self.model.eval()
        metrics.reset()
        epoch_logs = {}

        if verbose > 0:
            progress.on_train_begin({})
            progress.on_epoch_begin(1, epoch_logs)

        loss_sum = 0
        batch_logs = {}
        for batch, batch_data in zip(range(n_steps), data_loader):
            if verbose > 0:
                progress.on_batch_begin(batch, batch_logs)

            output = self.evaluate_batch(*batch_data, device)
            if self.loss_fn is not None:
                loss, output = output
                loss_sum += loss.item()
                batch_logs['loss'] = loss_sum / (batch + 1)
            metrics.update(output)

            if verbose > 0:
                progress.on_batch_end(batch, batch_logs)

        res = metrics.compute()

        if self.loss_fn is not None:
            _key = 'val:loss'
            loss = loss_sum / len(data_loader)
            epoch_logs[_key] = loss
            res['val:loss'] = loss

        if verbose > 0:
            res = _check_progbar_logger_value('', res)
            epoch_logs.update(res)

        if verbose > 0:
            progress.on_epoch_end(1, epoch_logs)
            progress.on_train_end(epoch_logs)

        return res

    def stop_training(self):
        self._stop_training = True

    def save_weights(self, filepath, overwrite=True):
        torch.save(self.model.state_dict(), filepath)

    def save(self, filepath, overwrite=True):
        torch.save(self.model, filepath)

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)








