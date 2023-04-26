# Another example is here, but it seems to be difficult: https://gist.github.com/sam-writer/723baf81c501d9d24c6955f201d86bbb
# I took this one: https://github.com/paulthemagno/transformers/commit/12b33f40c8fa29938ac6c8c2bcfd5e9dd224a911


from typing import Optional, Tuple, Union
from transformers import MT5Config, MT5Model, PreTrainedModel, MT5ForConditionalGeneration, MT5EncoderModel

from transformers.modeling_outputs import (
    SequenceClassifierOutput, TokenClassifierOutput
)
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), './'))

# Huggingface has the exact code in their Github repro, but for some reasons you cannot access it through the API: https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/mt5/modeling_mt5.py#L1291
from modeling_mt5 import MT5PreTrainedModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss


# I inserted this class here to pool the output,it's a copy of BertPooler
class MT5Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MT5PoolerForHier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # IMPORTANT: Remove this, because we already do pooling in hierbert
        # first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# copied by BertForSequenceClassification
class MT5ForSequenceClassification(MT5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.mt5 = MT5EncoderModel(config)
        # classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        # AttributeError: 'MT5Config' object has no attribute 'classifier_dropout' and not attribute 'hidden_dropout_prob'
        classifier_dropout = config.dropout_rate

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # added not to raise error (this is not in BertForSequenceClassification), copied by MT5ForConditionalGeneration
        self.model_parallel = False

        # this should be in MT5Model, as it is in BertModel, but I inserted here not return different output in MT5Model
        # MT5Pooler is a copy of BertPooler
        # I coMented 'if add_pooling_layer else None' not to add additional parameters, but there is a 'add_pooling_layer' param in BertModel
        self.pooler = MT5Pooler(config)  # if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.mt5(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids, # it has no token_type_ids
            # position_ids=position_ids, # it has no position_ids
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

            # this is my main doubt: MT5Model needs boht input_ids/embeds and decoder_input_ids/emebds while BertModel only input_ids/embeds
            # I tried to pass to MT5Model the same values for these variables but I'm not sure about that
            # decoder_input_ids=input_ids,
            # decoder_inputs_embeds=inputs_embeds,
        )

        # added from BertModel (but I wanted to use here not to break MT5Model output)
        pooled_output = self.pooler(outputs[0]) if self.pooler is not None else None
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # outputs doesn't have hidden_states and attentions as in Bert
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
            # but it has the same variables with decoder_* at the beginning, since outputs is of the type Seq2SeqModelOutput
            # hidden_states=outputs.decoder_hidden_states,
            # attentions=outputs.decoder_attentions,
        )


class HierMT5ForSequenceClassification(MT5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.mt5 = MT5EncoderModel(config)
        # classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        # AttributeError: 'MT5Config' object has no attribute 'classifier_dropout' and not attribute 'hidden_dropout_prob'
        classifier_dropout = config.dropout_rate

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # added not to raise error (this is not in BertForSequenceClassification), copied by MT5ForConditionalGeneration
        self.model_parallel = False

        # this should be in MT5Model, as it is in BertModel, but I inserted here not return different output in MT5Model
        # MT5Pooler is a copy of BertPooler
        # I coMented 'if add_pooling_layer else None' not to add additional parameters, but there is a 'add_pooling_layer' param in BertModel
        self.pooler = MT5PoolerForHier(config)  # if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.mt5(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids, # it has no token_type_ids
            # position_ids=position_ids, # it has no position_ids
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

            # this is my main doubt: MT5Model needs boht input_ids/embeds and decoder_input_ids/emebds while BertModel only input_ids/embeds
            # I tried to pass to MT5Model the same values for these variables but I'm not sure about that
            # decoder_input_ids=input_ids,
            # decoder_inputs_embeds=inputs_embeds,
        )

        # added from BertModel (but I wanted to use here not to break MT5Model output)
        pooled_output = self.pooler(outputs[0]) if self.pooler is not None else None
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # outputs doesn't have hidden_states and attentions as in Bert
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
            # but it has the same variables with decoder_* at the beginning, since outputs is of the type Seq2SeqModelOutput
            # hidden_states=outputs.decoder_hidden_states,
            # attentions=outputs.decoder_attentions,
        )


# Taken from https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/bert/modeling_bert.py#L1714
class MT5ForTokenClassification(MT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = MT5EncoderModel(config)  # add_pooling_layer=False
        classifier_dropout = config.dropout_rate
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.model_parallel = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            # token_type_ids: Optional[torch.Tensor] = None,
            # position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
