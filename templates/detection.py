# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

import torch
from torch.utils.data import DataLoader

from thtrainer import callbacks, tensorboard
from thtrainer.metrics import coco
from thtrainer.trainer import Trainer


def make_transforms(train=True):
    '''You can refer to `https://github.com/jcoral/Transforms/blob/master/segmentation.py`'''
    transforms = [
        # TODO: Resize
    ]
    if train:
        # TODO: Augmente PIL image
        pass

    transforms.extend([
        T.ToTensor(),
    ])

    if train:
        # TODO: Augmente tensor
        pass

    # Noramalize
    transforms.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

    return T.Compose(transforms)


def make_dataset(use_aug=False):
    train_transform = make_transforms(train=use_aug)
    test_transform = make_transforms(train=False)

    # TODO: Imp dataset
    train_dataset = None
    test_dataset = None

    return train_dataset, test_dataset


def make_dataloader(train_dataset, test_dataset, batch_size=32,
                    train_shuffle=True, num_workers=0):
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=train_shuffle,
        collate_fn=coco.COCOMetric.collate_fn,
        num_workers=num_workers
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=coco.COCOMetric.collate_fn,
        num_workers=num_workers
    )

    return train_data_loader, test_data_loader


def make_callbacks(optim=None, model_path=None,
                   scheduler=None, monitor=None, patience=1,
                   tensorboard_logdir=None):
    used_callbacks = []

    # If you use the learning rate scheduler,
    # you will need to add LRSchedulerCallback
    if scheduler is not None:
        scheduler = callbacks.LRSchedulerCallback(scheduler(optim))
        used_callbacks.append(scheduler)

    # Use checkpoint
    if model_path is not None:
        checkpoint = callbacks.ModelCheckpoint(
            model_path, 'train:loss',
            save_best_only=True,
            period=10, verbose=1,
            mode='min',
            save_weights_only=True
        )
        used_callbacks.append(checkpoint)

    # Use early stopping
    if monitor is not None:
        early_stop = callbacks.EarlyStopping(monitor=monitor,
                                             patience=patience,
                                             mode='min',
                                             restore_best_weights=False)
        used_callbacks.append(early_stop)

    if tensorboard_logdir is not None:
        used_callbacks.append(tensorboard.TensorBoard(tensorboard_logdir))

    return used_callbacks


def train_on_batch(trainer):

    def _wrapper(images, y):
        images = images.to(trainer.device)
        trainer.optimizer.zero_grad()
        loss, out_bboxes, out_scores = trainer.model(images, y)
        trainer.metrics.update(((out_bboxes, out_scores), y))
        loss.backward()
        trainer.optimizer.step()
        return loss

    return _wrapper


def evaluate_on_batch(trainer):
    def _wrapper(images, y, device=None):
        images = images.to(device)
        trainer.model.eval()
        with torch.no_grad():
            output = trainer.model(images, y)
        return (output[0], (output[1:], y))

    return _wrapper


def train_trainer(img_size, lr=1e-4,
                  epochs=20, batch_size=32,
                  make_optim=None, make_scheduler=None,
                  load_model=False, model_path=None,
                  use_aug=False,
                  monitor=None, patience=1, verbose=1,
                  tensorboard_logdir=None, history_path=None,
                  validate_init=1, validate_steps=1
                 ):
    if not isinstance(img_size, (list, tuple)):
        img_size = (img_size, img_size)

    train_dataset, test_dataset = make_dataset(use_aug)
    train_data_loader, test_data_loader = make_dataloader(train_dataset, test_dataset,
                                                          batch_size=batch_size,
                                                          num_workers=batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO: Imp model
    model = None
    if load_model:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    model_params = [p for p in model.parameters() if p.requires_grad]
    if make_optim is None:
        optim = torch.optim.Adam(model_params, lr, weight_decay=1e-4)
    else:
        optim = make_optim(model_params)

    scheduler = None
    if make_scheduler is None:
        scheduler = make_scheduler(optim)

    used_callbacks = make_callbacks(optim=optim, model_path=model_path,
                          scheduler=scheduler, monitor=monitor, patience=patience,
                          tensorboard_logdir=tensorboard_logdir)

    trainer = Trainer(
        model,
        optim,
        loss_fn=None,
        callbacks=used_callbacks,
        device=device,
        metrics=[
            coco.COCOMetric(
                train_data_loader,
                img_size,
                iou_types=['bbox'],
                device=device,
                output_transform=coco.bbox_transform
            )
        ],
        val_metrics=[
            coco.COCOMetric(
                test_data_loader,
                img_size,
                iou_types=['bbox'],
                device=device,
                output_transform=coco.bbox_transform
            )
        ]
    )
    trainer.train_on_batch = train_on_batch(trainer)
    trainer.evaluate_batch = evaluate_on_batch(trainer)

    history = trainer.fit(
        train_data_loader,
        epochs=epochs,
        batch_size=batch_size,
        loss_log=True,
        validation_data=test_data_loader,
        verbose=verbose,
        validate_init=validate_init,
        validate_steps=validate_steps
    )


    if history_path:
        with open(history_path, 'w') as f:
            import json
            json.dump(history.history, f)

    return trainer


def main():
    lr = 1e-3
    epochs = 20
    def make_scheduler(optim):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs, 0)

    def make_optim(params):
        return torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    train_trainer(img_size=300, lr=lr,
                  epochs=100, batch_size=32,
                  make_optim=make_optim, make_scheduler=make_scheduler,
                  load_model=False, model_path=None,
                  use_aug=True,
                  monitor='val_loss', patience=1, verbose=1,
                  tensorboard_logdir=None, history_path=None)


if __name__ == '__main__':
    main()





