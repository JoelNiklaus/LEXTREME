from torch import nn
from transformers import add_start_docstrings
from transformers.activations import ACT2FN
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout, DEBERTA_START_DOCSTRING, \
    DebertaV2ForSequenceClassification, DebertaV2Model

# Copied from transformers.models.deberta.modeling_deberta.ContextPooler
class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        # IMPORTANT: Remove this, because we already do pooling in hierbert
        # context_token = hidden_states[:, 0]
        context_token = self.dropout(hidden_states)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size




@add_start_docstrings(
    """
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
# Copied from transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification with Deberta->DebertaV2
class HierDebertaV2ForSequenceClassification(DebertaV2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()
