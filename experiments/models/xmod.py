from transformers import XmodPreTrainedModel, XmodModel

import torch
from torch import nn
# from transformers.models.xmod.modeling_xmod import XmodClassificationHead
from transformers import XmodForSequenceClassification


# Copied directly from https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/xmod/modeling_xmod.py#L1546
class HierXmodClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # We need to remove this for hierarchical variants
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HierXmodForSequenceClassification(XmodForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification.__init__ with Roberta->Xmod
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XmodModel(config, add_pooling_layer=False)
        self.classifier = HierXmodClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()
