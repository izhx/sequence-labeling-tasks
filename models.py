"""

"""

from typing import Dict, List, Tuple, Union, Any, cast

import torch
from torch.nn import Embedding

# from allennlp.modules import ConditionalRandomField, TimeDistributed

from nmnlp.core import Model, Vocabulary
from nmnlp.modules.embedding import build_word_embedding
from nmnlp.modules.encoder import build_encoder
from nmnlp.modules.dropout import WordDropout
from nmnlp.modules.linear import NonLinear
from nmnlp.models.dependency_parser import remove_sep

from conditional_random_field import ConditionalRandomField


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
            self.embedding = lambda ids, seqs, **kwargs: self.word_embedding(
                seqs, **kwargs)
        else:
            self.embedding = lambda ids, seqs, **kwargs: self.word_embedding(
                ids, **kwargs)
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
        self.crf = ConditionalRandomField(len(vocab[label_namespace]))
        self.top_k = top_k

    def forward(self,
                indicator: torch.Tensor,
                upostag: torch.Tensor,
                words: torch.Tensor = None,
                sentences: List = None,
                mask: torch.Tensor = None,
                seq_lens: torch.Tensor = None,
                labels: torch.Tensor = None,
                **kwargs) -> Tuple[Union[torch.Tensor, List, Dict]]:
        feat = self.embedding(words, sentences, **kwargs)
        if self.word_transform is not None:
            feat = self.word_transform(feat)
        
        indicator = self.indicator_embedding(indicator)
        upostag = self.pos_embedding(upostag)
        feat = torch.cat([feat, upostag, indicator], dim=-1)


        feat = self.word_dropout(feat)
        if self.encoder is not None:
            feat = self.encoder(feat, seq_lens, **kwargs)  # unpack会去掉[SEP]那一列
            if feat.shape[1] == mask.shape[1] - 1:
                mask, labels = remove_sep([mask, labels])
        feat = self.word_dropout(feat)

        scores = self.tag_projection_layer(feat)
        output = {}

        if not self.training:
            best_paths = self.crf.viterbi_tags(scores, mask, top_k=self.top_k)
            # Just get the top tags and ignore the scores.
            predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])
            output['predicted_tags']

        if tags is not None:
            crf_mask = mask
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(scores, tags, crf_mask)
            output['loss'] = -log_likelihood

            if not self.training:
                # Represent viterbi tags as "class probabilities" that we can
                # feed into the metrics
                class_probabilities = scores * 0.0
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[i, j, tag_id] = 1

                # for metric in self.metrics.values():
                #     metric(class_probabilities, tags, mask)

        return output
