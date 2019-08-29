# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import absolute_import

import time

from thtrainer.metrics import Metric
import torch

from thtrainer.metrics.coco_lib.coco_eval import CocoEvaluator
from thtrainer.metrics.coco_lib.coco_utils import get_coco_api_from_dataset
from thtrainer.metrics.coco_lib import utils as det_utils


def bbox_transform(output, metric):
    '''
    if dataset is COCOMetricDataset and iou_type is bbox, use it

    # Arguments
        output: `tuple((bboxes, scores), target)`.
        metric: `COCOMetric`
        :return: (pred, targets)
    '''

    outputs, targets = output
    cpu_device = torch.device("cpu")
    out = []
    boxes, scores = outputs
    for b, s in zip(boxes, scores):
        b = det_utils.to_xyxy(b, metric.img_size[0], metric.img_size[1])
        labels = s
        max_score=s
        if len(s) != 0:
            max_score = s.max(-1).values
            labels = s.argmax(-1)

        out.append(dict(
            boxes=b,
            labels=labels,
            scores=max_score
        ))
    outputs = out
    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    return (outputs, targets)


class COCOMetricDataset(torch.utils.data.Dataset):

    N_CLASSES = 21

    def __init__(self, transforms=None):
        self.transforms = transforms

    def __getitem__(self, idx):
        # TODO: Read image
        # img = ...

        # TODO: Overwrite boxes, shape(k, 4)
        boxes = torch.randint(0, 200, (10, 4))

        boxes = torch.tensor(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = None # TODO: Make labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # TODO: Use transforms
        # if self.transforms is not None:
            # img, target = self.transforms(img, target)

        # TODO: Remove comment
        # return img, target

    def __len__(self):
        # TODO: Overwrite, return number of total samples
        return 1


class COCOMetric(Metric):
    '''
    # Arguments
        `update` must receive output of the form `(y_pred, y)`.
        `y_pred` must be in the following shape (batch_size, n, 4).
        `y` must be in the following shape (batch_size, ...).
        `y` and `y_pred` must be in the following shape of (batch_size, num_categories, ...) for multilabel cases.
    '''

    '''
    :param data_loader: torch DataLoader
    :param img_size: (h, w)
    :param iou_types: `bbox` or `segm` or `keypoints`, default `bbox`
    :param device: torch device
    '''

    bbox_eval_keys = [
       # AP:IoU:area:maxDets
        'AP:0.50-0.95:all:100',
        'AP:0.50:all:100',
        'AP:0.75:all:100',
        'AP:0.50-0.95:small:100',
        'AP:0.50-0.95:medium:100',
        'AP:0.50-0.95:large:100',
        'AR:0.50-0.95:all:1',
        'AR:0.50-0.95:all:10',
        'AR:0.50-0.95:all:100',
        'AR:0.50-0.95:small:100',
        'AR:0.50-0.95:medium:100',
        'AR:0.50-0.95:large:100',
    ]

    def __init__(self, data_loader,
                 img_size,
                 iou_types=None,
                 device=None,
                 output_transform=lambda x, metric: x):


        self.img_size = img_size
        self.iou_types = iou_types or ['bbox']
        self.output_transform = output_transform
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        coco = get_coco_api_from_dataset(data_loader.dataset)
        self.coco_evaluator = CocoEvaluator(coco, self.iou_types)
        self.evaluator_time = 0

        super(COCOMetric, self).__init__()

    @staticmethod
    def collate_fn(batch):
        '''
        if dataset is COCOMetricDataset, use it
        '''
        x = [b[0] for b in batch]
        target = [b[1] for b in batch]
        return torch.stack(x), target

    def reset(self):
        self.coco_evaluator.init()

    def update(self, output):
        outputs, targets = self.output_transform(output, self)

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        self.coco_evaluator.update(res)

    def compute(self):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        res = {}
        for k, v in self.coco_evaluator.coco_eval.items():
            stats = v.stats.tolist()
            res[k] = dict(zip(self.bbox_eval_keys, stats))
        return res
