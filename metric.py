from collections import OrderedDict

import torch


class SRLMetric(object):
    def __init__(self, ignore_index=0):
        self.counter = OrderedDict({'num': 0, 'got': 0, 'correct': 0})
        self.ignore_index = ignore_index
        self.best_metric = None

    def _update_counter(self, num, got, correct):
        self.counter['num'] += num
        self.counter['got'] += got
        self.counter['correct'] += correct

    def __call__(
        self,  # type: ignore
        indicator: torch.Tensor = None,
        mask: torch.Tensor = None,
        predicted_tags: torch.Tensor = None,
        gold_tags: torch.Tensor = None,
        reset: bool = False):

        if reset:
            metrics = self.get_metric(*self.counter.values())
            self.counter = OrderedDict({k: 0 for k in self.counter})
            return metrics

        mask = (gold_tags != self.ignore_index).long() * (mask - indicator) # 只看标注
        num = mask.sum().item()
        got = ((predicted_tags != self.ignore_index).long() * mask).sum().item()
        correct = ((predicted_tags == gold_tags).long() * mask).sum().item()
        self._update_counter(num, got, correct)

        return self.get_metric(num, got, correct)

    def get_metric(self, num, got, correct):
        # recal, 答案被召回几个
        # pre  预测对了几个
        recall = got / num
        precision = correct / num
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return {'F1': f1, 'precision': precision, 'recall': recall}

    def is_best(self, metric):
        if self.best_metric is None:
            self.best_metric = metric
            return True
        return metric['F1'] > self.best_metric['F1']


class DependencyParsingMetric(object):
    def __init__(self, ignore_index=0):
        self.counter = OrderedDict({'num': 0, 'arc': 0, 'rel': 0})
        self.best_metric = None

    def __call__(self,
                 head_pred: torch.Tensor = None,
                 rel_pred: torch.Tensor = None,
                 head_gt: torch.Tensor = None,
                 rel_gt: torch.Tensor = None,
                 mask: torch.Tensor = None,
                 reset: bool = False):

        if reset:
            metrics = self.get_metric(*self.counter.values())
            self.counter = OrderedDict({k: 0 for k in self.counter})
            return metrics

        if len(rel_pred.shape) > len(rel_gt.shape):
            rel_pred = rel_pred.max(2)[1]

        mask[:, 0] = 0  # mask out <root> tag
        head_pred_correct = (head_pred == head_gt).long() * mask
        rel_pred_correct = (rel_pred == rel_gt).long() * head_pred_correct
        arc = head_pred_correct.sum().item()
        rel = rel_pred_correct.sum().item()
        num = mask.sum().item()

        self.counter['arc'] += arc
        self.counter['rel'] += rel
        self.counter['num'] += num

        return self.get_metric(num, arc, rel)

    def get_metric(self, num, arc, rel):
        return {'UAS': arc * 1.0 / num, 'LAS': rel * 1.0 / num}

    def is_best(self, metric):
        if self.best_metric is None:
            self.best_metric = metric
            return True
        if metric['UAS'] > self.best_metric['UAS']:
            return True
        elif metric['UAS'] == self.best_metric['UAS']:
            return metric['LAS'] > self.best_metric['LAS']
        else:
            return False
