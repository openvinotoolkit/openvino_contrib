import tensorflow as tf
import numpy as np
from .base_wrapper import BaseWrapper

def iou_score(pred, target, num_classes):
    pred = pred.argmax(axis=-1)  # (batch, H, W)
    target = np.array(target).astype(int)
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

class SegmentationWrapper(BaseWrapper):
    """Wrapper for semantic segmentation tasks."""

    def evaluate_iou(self, x, y, num_classes, batch_size=32):
        """Evaluate mean IoU score."""
        preds = self.model.predict(x, batch_size=batch_size)
        return iou_score(preds, y, num_classes)

    def evaluate_dice(self, x, y, num_classes, batch_size=32):
        """Evaluate mean Dice score."""
        preds = self.model.predict(x, batch_size=batch_size)
        pred = preds.argmax(axis=-1)
        target = np.array(y).astype(int)
        dices = []
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            intersection = np.logical_and(pred_cls, target_cls).sum()
            dice = (2. * intersection) / (pred_cls.sum() + target_cls.sum() + 1e-8)
            dices.append(dice)
        return np.nanmean(dices)