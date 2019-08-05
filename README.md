# Torch trainer
The Torch trainer, and similar keras, callbacks from keras, has the same API with keras.

# Example

```
data_loader = DataLoader(dataset, 10, True, collate_fn=collate_fn, num_workers=4)

optim = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
scheduler = LRSchedulerCallback(StepLR(optim, step_size=4, gamma=0.5))
loss_fn = torch.nn.MSELoss(reduction='sum')

trainer = Trainer(model, optim, loss_fn, callbacks=[scheduler])
trainer.train_on_batch = train_on_batch(trainer) # Custom train_on_batch
trainer.fit(data_loader, epochs=50)
```
