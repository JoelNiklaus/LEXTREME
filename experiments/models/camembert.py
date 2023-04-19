from transformers import add_start_docstrings, CamembertConfig
from transformers.models.camembert.modeling_camembert import CAMEMBERT_START_DOCSTRING

from models.roberta import HierRobertaForSequenceClassification


@add_start_docstrings(
    """
    CamemBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
class HierCamembertForSequenceClassification(HierRobertaForSequenceClassification):
    """
    This class overrides [`HierRobertaForSequenceClassification`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = CamembertConfig
