## thtrainer.Trainer

```
Trainer(model, optimizer, loss_fn, callbacks=None, metrics=None, val_metrics=None, device=None)
```

*Arguments*
- model: `Module`.
- optimizer: `Optimizer`.
- loss_fn: `Module`, Compute model loss
- callbacks: List of `thtrainer.callbacks.Callback` instances, List of callbacks to apply during training.
- metrics: List of `thtrainer.metrics.Metric` instances, Compute metrics at the end of each batch.
- val_metrics: List of `thtrainer.metrics.Metric` instances, Evaluate validation data use val_metrics. if val_metrics is `None`: use metrics

<br/>
*Example*

```python 
data_loader = DataLoader(dataset, 10, True, collate_fn=collate_fn, num_workers=4)

optim = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
scheduler = LRSchedulerCallback(StepLR(optim, step_size=4, gamma=0.5))

trainer = Trainer(model, optim,None, callbacks=[scheduler])
trainer.train_on_batch = train_on_batch(trainer) # Custom train_on_batch
trainer.fit(data_loader, epochs=50)
```
<br/>
### Methods
#### fit
Train model.
```
fit(self,data_loader,epochs=1,batch_size=32,verbose=1,validation_data=None,loss_log=None,shuffle=True,validate_steps=1, validate_init=1)
```

*Arguments*
- batch_size: Integer, Number of samples per gradient update. If unspecified, batch_size will default to 32.
- epochs: Integer, Number of epochs to train the model.
- verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
- validation_data: `DataLoader` or `Dataset`, Evaluate validation_data the loss and any model - metrics at the end of each epoch.
    The model will not be trained on this data.
- shuffle: Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
- validate_steps: Integer, Evaluate validation data when epoch % validate_steps == 0
- validate_init: Integer, Evaluate validation data when epoch == validate_init

<br/>
#### evaluate
Evaludate model.
```
evaluate(self,data_loader, batch_size=1,shuffle=False, metrics=None,verbose=1, device=None,loss_log=False)
```


*Arguments*
- data_loader: `DataLoader`， batch_size: `Integer` DataLoader batch size
- shuffle: Boolean (whether to shuffle the training data
    before each epoch) or str (for 'batch').
- metrics: List of `thtrainer.metrics.Metric` instances, Compute metrics at the end of each batch.
- verbose: `Integer`. 0, 1, or 2. Verbosity mode, 0 = silent, 1 = progress bar, 2 = one line per epoch.
- loss_log: `Bool`, train batch whether return loss, if `True` or loss function is not None: Progbar logger show loss, if `False` and loss function is None:  Progbar logger not show loss

<br/>
#### stop_training
Stoping train model.
```
stop_training(self)
```
<br/>

#### save_weights
Save model params.
```
save_weights(self, filepath, overwrite=True)
```
<br/>

#### save
Save model.

```
save(self, filepath, overwrite=True)torch.save(self.model, filepath)
```
<br/>

#### get_weights
Get model params
```
get_weights(self)
```
<br/>

#### set_weights
Set mdoel params
```
set_weights(self, weights)
```
<br/><br/>
## Metrics
### Accuracy
Calculates the accuracy for binary, multiclass and multilabel data.

*Arguments*
- output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
- is_multilabel (bool, optional): flag to use in multilabel case. By default, False.

<br/>
*update*
update must receive output of the form `(y_pred, y)`.In binary and multilabel cases, the elements of `y` and `y_pred` should have 0 or 1 values.
- `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
- `y` must be in the following shape (batch_size, ...).
- `y` and `y_pred` must be in the following shape of (batch_size, num_categories, ...) for multilabel cases.

### Loss
Calculates the average loss according to the passed loss_fn.

*Arguments*
- loss_fn (callable): a callable taking a prediction tensor, a target
    tensor, optionally other arguments, and returns the average loss
    over all observations in the batch.
- output_transform (callable): a callable that is used to transform the
    :class:`~ignite.engine.Engine`'s `process_function`'s output into the
    form expected by the metric.
    This can be useful if, for example, you have a multi-output model and
    you want to compute the metric with respect to one of the outputs.
    The output is is expected to be a tuple (prediction, target) or
    (prediction, target, kwargs) where kwargs is a dictionary of extra
    keywords arguments.
- batch_size (callable): a callable taking a target tensor that returns the
    first dimension size (usually the batch size).
    
### MeanAbsoluteError
Calculates the mean absolute error.
### MeanPairwiseDistance
Calculates the mean pairwise distance.
### MeanSquaredError
Calculates the mean squared error.
### Precision
Calculates precision for binary and multiclass data.
### Recall
Calculates recall for binary and multiclass data.
### RootMeanSquaredError
Calculates the root mean squared error.
### TopKCategoricalAccuracy
Calculates the top-k categorical accuracy.
### ConfusionMatrix
Calculates confusion matrix for multi-class data.
### MetricList
Metric list.
### IOUMetric
Calculates Intersection over Union
### MeanIOUMetric
Calculates mean Intersection over Union

### COCOmAPMetric
Calculates mAP.
Include:
AP:0.50-0.95:all:100,
AP:0.50:all:100,
AP:0.75:all:100, 
AP:0.50-0.95:small:100, 
AP:0.50-0.95:medium:100,
AP:0.50-0.95:large:100, 
AR:0.50-0.95:all:1,
AR:0.50-0.95:all:10,
AR:0.50-0.95:all:100,
AR:0.50-0.95:small:100,
AR:0.50-0.95:medium:100,
AR:0.50-0.95:large:100,

*update arguments*
- update must receive output of the form `(y_pred, y)`.
- y_pred must be in the following shape (batch_size, n, 4).
- y must be in the following shape (batch_size, ...).
- y and y_pred must be in the following shape of (batch_size, num_categories, ...) for multilabel cases.

*Example*

```
dataset = Dataset() # dataset from thtrainer.metrics.coco.COCOMetricDataset
data_loader = DataLoader(dataset, 16, shuffle=False, collate_fn=coco.COCOMetric.collate_fn)
metric = coco.COCOMetric(data_loader,
                         img_size=(300, 300),
                         output_transform=coco.bbox_transform)
```

<br/><br/>
### Callback
see [keras](https://keras.io/callbacks/)