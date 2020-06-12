"""

"""

from typing import Dict, List, Tuple, Union, Any, cast

import torch
from torch.nn import Embedding

from allennlp.modules import ConditionalRandomField, TimeDistributed

from nmnlp.core import Model, Vocabulary
from nmnlp.modules.embedding import build_word_embedding
from nmnlp.modules.encoder import build_encoder
from nmnlp.modules.dropout import WordDropout
from nmnlp.modules.linear import NonLinear

# from conditional_random_field import ConditionalRandomField
from metric import SRLMetric


def build_model(name, **kwargs):
    m = {
        'srl': SemanticRoleLabeler,
    }
    return m[name](**kwargs)


class SemanticRoleLabeler(Model):
    """
    a
    """
    def __init__(self,
                 vocab: Vocabulary,
                 word_embedding: Dict[str, Any],
                 transform_dim: int = 0,
                 pos_dim: int = 50,
                 indicator_dim: int = 50,
                 encoder: Dict[str, Any] = None,
                 dropout: float = 0.33,
                 label_namespace: str = "labels",
                 top_k: int = 1) -> None:
        super().__init__()
        self.word_embedding = build_word_embedding(num_embeddings=len(vocab['words']),
                                                   vocab=vocab,
                                                   dropout=dropout,
                                                   **word_embedding)
        if 'elmo' in word_embedding['name_or_path']:
            self.embedding = lambda ids, seqs, mask, **kwargs: self.word_embedding(
                seqs, **kwargs)
        else:
            self.embedding = lambda ids, seqs, mask, **kwargs: self.word_embedding(
                ids, mask=mask, **kwargs)
        feat_dim: int = self.word_embedding.output_dim

        if transform_dim > 0:
            self.word_transform = NonLinear(feat_dim, transform_dim)
            feat_dim: int = transform_dim
        else:
            self.word_transform = None

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
            return self.metric.get_metric(counter.values())
        else:
            return self.metric(reset=reset)

    def forward(self,
                indicator: torch.Tensor,
                upostag: torch.Tensor,
                words: torch.Tensor = None,
                sentences: List = None,
                mask: torch.Tensor = None,
                seq_lens: torch.Tensor = None,
                labels: torch.Tensor = None,
                **kwargs) -> Tuple[Union[torch.Tensor, List, Dict]]:
        feat = self.embedding(words, sentences, mask, **kwargs)
        if self.word_transform is not None:
            feat = self.word_transform(feat)

        emb = self.indicator_embedding(indicator), self.pos_embedding(upostag)
        feat = torch.cat([feat, *emb], dim=-1)

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
