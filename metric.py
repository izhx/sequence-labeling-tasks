from collections import OrderedDict

import torch

class SRLMetric(object):
    def __init__(self, ignore_index=0):
        self.counter = OrderedDict({'num': 0, 'got': 0, 'correct': 0})
        self.ignore_index = ignore_index
    
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
        return {'F1': f1, 'precision':precision, 'recall': recall}
