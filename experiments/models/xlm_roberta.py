from transformers import XLMRobertaConfig, add_start_docstrings
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLM_ROBERTA_START_DOCSTRING

from models.roberta import HierRobertaForSequenceClassification


@add_start_docstrings(
    """
    XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class HierXLMRobertaForSequenceClassification(HierRobertaForSequenceClassification):
    """
    This class overrides [`HierRobertaForSequenceClassification`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
