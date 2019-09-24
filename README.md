[TOC]

# Torch trainer
The Torch trainer, and similar keras, callbacks from keras, has the same API with keras. see: [keras](https://keras.io/zh/)

Some of metrics from ignite, see: [ignite](https://pytorch.org/ignite)

Torch trainer api: [API](https://github.com/jcoral/Torch_trainer/blob/master/API.md)

# 1. Base example: Using trainer to train model

## 1.1、Define model and dataset
```python
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
```

## 1.2. Train model
```python
# init model, optim, loss_fn, dataset
model = TestModel()
optim = torch.optim.Adam(self.model.parameters())
loss_fn = nn.CrossEntropyLoss()
ds = DS()

# init trainer
trainer = Trainer(
    model, optim,
    loss_fn
)

# train
trainer.fit(ds, epochs=1, verbose=1)
```

# 2. Use metrics
if you use metrics, you need to install ignite, [ignite document](https://pytorch.org/ignite)
Metric is defined in `thtrainer.metrics`, and support string, for: `accuracy`, `loss'`, `top-k`, `mae`, `mse`.

```python
from thtrainer.metrics import Accuracy, TopKCategoricalAccuracy

trainer = Trainer(
    model, optim,
    loss_fn,
    metrics=[
        Accuracy(),
        TopKCategoricalAccuracy(),
        'loss' # use string
    ])

history = trainer.fit(ds, batch_size=2,
                      epochs=10, verbose=1,
                      validation_data=ds)
print(history.history)
```

## 2.1. Write your metric

```python
from thtrainer.metrics import Metric

# TODO: implement these methods
class MyMetric(Metric):

    def reset(self):
        """
        Resets the metric to it's initial state.
        This is called at the start of each epoch.
        """
        pass

    def update(self, output):
        """
        Updates the metric's state using the passed batch output.
        This is called once for each batch.

        Args:
            output: the is the output from the engine's process function.
        """
        pass

    def compute(self):
        """
        Computes the metric based on it's accumulated state.
        This is called at the end of each epoch.

        Returns:
            Any: the actual quantity of interest.

        Raises:
            NotComputableError: raised when the metric cannot be computed.
        """
        pass
```

## 2.2. Evaluate COCO dectection result
If you use COCOMetric, please make sure you have installed `pycocotools`.
Install command: `pip install pycocotools`
see: `Torch_trainer/test/test_COCOMetric.py`

# 3. Callbacks

## 3.1. Use tensorboard
In section, you need to install tensorboard and tensorflow, see: [install tutorial](https://pytorch.org/docs/stable/tensorboard.html)
```python
ipt = torch.stack([ds[i][0] for i in range(2)])
trainer = Trainer(
    model, optim,
    loss_fn,
    callbacks=[TensorBoard('./runs_logs',
                           comment='Test',
                           input_to_model=ipt)],
    metrics=[
        Accuracy(),
        TopKCategoricalAccuracy(),
        'loss'
    ]
)

history = trainer.fit(
    ds, batch_size=2,
    epochs=10, verbose=1,
    validation_data=ds
)
```

## 3.2. Control the training process
In section, you can control the training process, for example: early stopping, save model, ...
```python
trainer = Trainer(
    model, optim,
    loss_fn,
    [
        TerminateOnNaN(),
        ModelCheckpoint('./tmp/model.pth', monitor='loss'),
        EarlyStopping(monitor='loss', verbose=1),
    ],
)

trainer.fit(ds, epochs=10, verbose=1)
```

## 3.3. Write your callback
It is the same as Keras.

```python
# TODO: To Implement some methods you want
class MyCallback(Callback):
    """Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.
    """

    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass
```


# 4. Write your train event

## 4.1. Write your train batch
```python
def train_on_batch(trainer):
    def _wrapper(X, y):
        images = images.to(trainer.device)
        trainer.optimizer.zero_grad()
        out = trainer.model(X)
        loss = trainer.loss_fn(out, y)
        trainer.metrics.update((out, y)) # if have metrics
        loss.backward()
        trainer.optimizer.step()
        return loss

    return _wrapper
    
trainer.train_on_batch = train_on_batch(trainer)
```

## 4.2. Write your evaluate batch

```python
def evaluate_on_batch(trainer):
    def _wrapper(images, y, device=None):
        images = images.to(device)
        output = trainer.model(images)
        loss = trainer.loss_fn(output, y)
        return (loss, output)

    return _wrapper

trainer.evaluate_batch = evaluate_on_batch(trainer)
```
