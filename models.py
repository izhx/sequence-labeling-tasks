"""

"""

from typing import Callable, Dict, List, Tuple, Union, Any, cast
from itertools import chain

import torch
from torch.nn import Embedding, ModuleList

from allennlp.modules import ConditionalRandomField, ScalarMix

from nmnlp.core import Model, Vocabulary
from nmnlp.models.dependency_parser import loss, DependencyParser, remove_sep
from nmnlp.modules.embedding import build_word_embedding
from nmnlp.modules.embedding.dep_embedding import DepSAWR
from nmnlp.modules.encoder import build_encoder
from nmnlp.modules.dropout import WordDropout
from nmnlp.modules.linear import NonLinear, Biaffine
from nmnlp.nn.chu_liu_edmonds import batch_decode_head

# from conditional_random_field import ConditionalRandomField
from metric import SRLMetric, TaggingMetric, DependencyParsingMetric


def build_model(name, **kwargs):
    m = {
        'srl': SemanticRoleLabeler,
        'dep': DepParser,
        'crf': CRFTagger
    }
    return m[name](**kwargs)


class SemanticRoleLabeler(Model):
    """
    a
    """
    def __init__(self,
                 vocab: Vocabulary,
                 word_embedding: Dict[str, Any],
                 depsawr: torch.nn.Module = None,
                 transform_dim: int = 0,
                 pos_dim: int = 50,
                 indicator_dim: int = 50,
                 encoder: Dict[str, Any] = None,
                 dropout: float = 0.33,
                 label_namespace: str = "labels",
                 top_k: int = 1,
                 **kwargs) -> None:
        super().__init__()
        self.word_embedding = build_word_embedding(num_embeddings=len(vocab['words']),
                                                   vocab=vocab,
                                                   dropout=dropout,
                                                   **word_embedding)
        feat_dim: int = self.word_embedding.output_dim

        if transform_dim > 0:
            self.word_transform = NonLinear(feat_dim, transform_dim)
            feat_dim: int = transform_dim
        else:
            self.word_transform = None

        if depsawr:
            dep_dim = kwargs.pop('dep_dim', 300)
            self.depsawr_forward = depsawr.forward
            self.projections = ModuleList(
                [NonLinear(i, dep_dim) for i in depsawr.dims])
            self.depsawr_mix = ScalarMix(len(depsawr.dims), True)
            feat_dim += dep_dim
        else:
            self.depsawr_forward = None

        self.pos_embedding = Embedding(len(vocab['upostag']), pos_dim, 0)
        self.indicator_embedding = Embedding(2, indicator_dim)
        feat_dim += (pos_dim + indicator_dim)

        if encoder is not None:
            self.encoder = build_encoder(feat_dim, dropout=dropout, **encoder)
            feat_dim = self.encoder.output_dim
        else:
            self.encoder = None

        self.tag_projection_layer = torch.nn.Linear(feat_dim, len(vocab[label_namespace]))
        self.word_dropout = WordDropout(dropout)
        self.crf = ConditionalRandomField(len(vocab[label_namespace]),
                                          include_start_end_transitions=False)
        self.top_k = top_k
        self.metric = SRLMetric(vocab[label_namespace]['_'])

    def get_metrics(self, counter=None, reset=False):
        if counter:
            return self.metric.get_metric(*counter.values())
        else:
            return self.metric(reset=reset)

    def forward(self,
                indicator: torch.Tensor,
                upostag: torch.Tensor,
                words: torch.Tensor = None,
                mask: torch.Tensor = None,
                seq_lens: torch.Tensor = None,
                labels: torch.Tensor = None,
                **kwargs) -> Tuple[Union[torch.Tensor, List, Dict]]:
        feat = self.word_embedding(words, mask=mask, **kwargs)
        if self.word_transform is not None:
            feat = self.word_transform(feat)

        embs = [self.indicator_embedding(indicator), self.pos_embedding(upostag)]
        if self.depsawr_forward is not None:
            dep_emb = self.depsawr_mix(self.depsawr_forward(
                kwargs['dw'], kwargs['ew'], mask, self.projections), mask)
            embs.append(dep_emb)
        feat = torch.cat([feat, *embs], dim=-1)

        feat = self.word_dropout(feat)
        if self.encoder is not None:
            feat = self.encoder(feat, seq_lens, **kwargs)
        feat = self.word_dropout(feat)

        scores = self.tag_projection_layer(feat)
        output = {}

        if not self.training:
            best_paths = self.crf.viterbi_tags(scores, mask, top_k=self.top_k)
            # Just get the top tags and ignore the scores.
            predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])
            output['predicted_tags'] = predicted_tags

        if labels is not None:
            # Add negative log-likelihood as loss
            output['loss'] = -self.crf(scores, labels, mask.bool())

            if not self.training:
                predicted = torch.zeros_like(labels)
                for i, tags in enumerate(predicted_tags):
                    predicted[i, :len(tags)] = torch.tensor(tags, dtype=torch.long, device=labels.device)
                output['metric'] = self.metric(indicator, mask, predicted, labels)

        return output

    @staticmethod
    def is_best(metric: Dict[str, float], former: Dict[str, float]) -> bool:
        return metric['F1'] > former['F1']


class CRFTagger(Model):
    """
    a
    """
    def __init__(self,
                 vocab: Vocabulary,
                 word_embedding: Dict[str, Any],
                 depsawr: torch.nn.Module = None,
                 transform_dim: int = 0,
                 pos_dim: int = 50,
                 encoder: Dict[str, Any] = None,
                 dropout: float = 0.33,
                 label_namespace: str = "labels",
                 top_k: int = 1,
                 **kwargs) -> None:
        super().__init__()
        self.word_embedding = build_word_embedding(num_embeddings=len(vocab['words']),
                                                   vocab=vocab,
                                                   dropout=dropout,
                                                   **word_embedding)
        feat_dim: int = self.word_embedding.output_dim

        if transform_dim > 0:
            self.word_transform = NonLinear(feat_dim, transform_dim)
            feat_dim: int = transform_dim
        else:
            self.word_transform = None

        if depsawr:
            dep_dim = kwargs.pop('dep_dim', 300)
            self.depsawr_forward = depsawr.forward
            self.projections = ModuleList(
                [NonLinear(i, dep_dim) for i in depsawr.dims])
            self.depsawr_mix = ScalarMix(len(depsawr.dims), True)
            feat_dim += dep_dim
        else:
            self.depsawr_forward = None

        self.pos_embedding = Embedding(len(vocab['upostag']), pos_dim, 0)
        feat_dim += pos_dim

        if encoder is not None:
            self.encoder = build_encoder(feat_dim, dropout=dropout, **encoder)
            feat_dim = self.encoder.output_dim
        else:
            self.encoder = None

        self.tag_projection_layer = torch.nn.Linear(feat_dim, len(vocab[label_namespace]))
        self.word_dropout = WordDropout(dropout)
        self.crf = ConditionalRandomField(len(vocab[label_namespace]),
                                          include_start_end_transitions=False)
        self.top_k = top_k
        self.metric = TaggingMetric(vocab[label_namespace]['_'])

    def get_metrics(self, counter=None, reset=False):
        if counter:
            return self.metric.get_metric(*counter.values())
        else:
            return self.metric(reset=reset)

    def forward(self,
                upostag: torch.Tensor,
                words: torch.Tensor = None,
                mask: torch.Tensor = None,
                seq_lens: torch.Tensor = None,
                labels: torch.Tensor = None,
                **kwargs) -> Tuple[Union[torch.Tensor, List, Dict]]:
        feat = self.word_embedding(words, mask=mask, **kwargs)
        if self.word_transform is not None:
            feat = self.word_transform(feat)

        embs = [self.pos_embedding(upostag)]
        if self.depsawr_forward is not None:
            dep_emb = self.depsawr_mix(self.depsawr_forward(
                kwargs['dw'], kwargs['ew'], mask, self.projections), mask)
            embs.append(dep_emb)
        feat = torch.cat([feat, *embs], dim=-1)

        feat = self.word_dropout(feat)
        if self.encoder is not None:
            feat = self.encoder(feat, seq_lens, **kwargs)
        feat = self.word_dropout(feat)

        scores = self.tag_projection_layer(feat)
        output = {}

        if not self.training:
            best_paths = self.crf.viterbi_tags(scores, mask, top_k=self.top_k)
            # Just get the top tags and ignore the scores.
            predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])
            output['predicted_tags'] = predicted_tags

        if labels is not None:
            # Add negative log-likelihood as loss
            output['loss'] = -self.crf(scores, labels, mask.bool())

            if not self.training:
                predicted = torch.zeros_like(labels)
                for i, tags in enumerate(predicted_tags):
                    predicted[i, :len(tags)] = torch.tensor(tags, dtype=torch.long, device=labels.device)
                output['metric'] = self.metric(mask, predicted, labels)

        return output

    @staticmethod
    def is_best(metric: Dict[str, float], former: Dict[str, float]) -> bool:
        return metric['F1'] > former['F1']


class DepParser(Model):
    """ """
    def __init__(self,
                 vocab: Vocabulary,
                 upostag: List[str],
                 word_embedding: Dict[str, Any],
                 depsawr: Dict[str, Any],
                 arc_dim: int,
                 rel_dim: int,
                 dropout: float = 0.5):
        super().__init__()
        self.word_embedding = build_word_embedding(
            num_embeddings=len(vocab['words']), vocab=vocab, **word_embedding)
        self.depsawr = DepSAWR(
            word_dim=self.word_embedding.output_dim, upostag=upostag,
            arc_dim=arc_dim, rel_dim=rel_dim, dropout=dropout, **depsawr)
        self.arc_classifier = Biaffine(arc_dim, arc_dim, 1)
        self.rel_classifier = Biaffine(rel_dim, rel_dim, len(vocab['deprel']))

        self.split_sizes = [arc_dim, rel_dim]
        self.metric = DependencyParsingMetric()
        self.is_best = DependencyParser.is_best

    def forward(self,
                words: torch.Tensor,
                upostag: torch.Tensor,
                mask: torch.Tensor,  # 有词的地方为1
                head: torch.Tensor = None,
                deprel: torch.Tensor = None,
                seq_lens: torch.Tensor = None,
                **kwargs):
        feat = self.word_embedding(words, mask=mask, **kwargs)
        feat = self.depsawr(feat, upostag, seq_lens)
        if feat[0].shape[1] == words.shape[1] - 1:
            mask, head, deprel = remove_sep([mask, head, deprel])

        feat = [f.split(self.split_sizes, dim=2) for f in feat]

        arc_pred = self.arc_classifier(feat[0][0], feat[1][0]).squeeze(-1)  # (b,s,s)
        rel_pred = self.rel_classifier(feat[0][1], feat[1][1])  # (b,s,s,c)

        head_pred = head if self.training else batch_decode_head(arc_pred, seq_lens)

        rel_pred = torch.gather(rel_pred, 2, head_pred.unsqueeze(
            2).unsqueeze(3).expand(-1, -1, -1, rel_pred.shape[-1])).squeeze(2)
        output = {'head_pred': head_pred, 'rel_pred': rel_pred}

        if head is not None:
            output['loss'] = loss(arc_pred, rel_pred, head, deprel, mask)
            if not self.training:
                output['metric'] = self.metric(
                    head_pred, rel_pred, head, deprel, mask)

        return output

    def get_metrics(self, counter=None, reset=False):
        if counter:
            return self.metric.get_metric(*counter.values())
        else:
            return self.metric(reset=reset)
