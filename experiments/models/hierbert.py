from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import numpy as np
from torch import nn
from transformers.file_utils import ModelOutput
from transformers import AutoModelForSequenceClassification, AutoTokenizer, MT5EncoderModel
from models.deberta_v2 import HierDebertaV2ForSequenceClassification
from models.distilbert import HierDistilBertForSequenceClassification
from models.roberta import HierRobertaForSequenceClassification
from models.xlm_roberta import HierXLMRobertaForSequenceClassification
from models.camembert import HierCamembertForSequenceClassification
from models.t5_encoder_classifier import T5ForSequenceClassification, HierT5ForSequenceClassification


@dataclass
class SimpleOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


def sinusoidal_init(num_embeddings: int, embedding_dim: int):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


supported_models = ['bert', 'distilbert', 'roberta', 'xlm-roberta', 'deberta-v2', 'camembert', 'mt5']


class HierarchicalBert(nn.Module):

    def __init__(self, encoder, max_segments=64, max_segment_length=128):
        super(HierarchicalBert, self).__init__()

        assert encoder.config.model_type in supported_models  # other model types are not supported so far
        # Pre-trained segment (token-wise) encoder, e.g., BERT
        self.encoder = encoder
        # Specs for the segment-wise encoder
        self.hidden_size = encoder.config.hidden_size
        self.max_segments = max_segments
        self.max_segment_length = max_segment_length

        # Init sinusoidal positional embeddings
        self.seg_pos_embeddings = nn.Embedding(max_segments + 1,
                                               encoder.config.hidden_size,
                                               padding_idx=0,
                                               _weight=sinusoidal_init(max_segments + 1,
                                                                       encoder.config.hidden_size))

        # Init segment-wise transformer-based encoder
        if encoder.config.model_type in ['distilbert']:
            # for some reason, intermediate_size is called differently in DistilBertConfig
            dim_feedforward = encoder.config.hidden_dim
            # for some reason, hidden_act is called activation in DistilBertConfig
            activation = encoder.config.activation
            # for some reason, hidden_dropout_prob is called dropout in DistilBertConfig
            dropout = encoder.config.dropout
            layer_norm_eps = 1e-5  # 1e-5 is default
        elif encoder.config.model_type in ['mt5']:
            # for some reason, intermediate_size is called differently in DistilBertConfig
            dim_feedforward = encoder.config.hidden_size
            # for some reason, hidden_act is called activation in DistilBertConfig
            activation = encoder.config.feed_forward_proj
            # for some reason, hidden_dropout_prob is called dropout in DistilBertConfig
            dropout = encoder.config.dropout_rate
            layer_norm_eps = 1e-5  # 1e-5 is default
        else:
            dim_feedforward = encoder.config.intermediate_size
            activation = encoder.config.hidden_act
            dropout = encoder.config.hidden_dropout_prob
            layer_norm_eps = encoder.config.layer_norm_eps

        if encoder.config.model_type in ['mt5']:
            self.seg_encoder = nn.Transformer(d_model=encoder.config.hidden_size,
                                              nhead=8,
                                              batch_first=True,
                                              dim_feedforward=dim_feedforward,
                                              activation='gelu',
                                              dropout=dropout,
                                              layer_norm_eps=layer_norm_eps,
                                              num_encoder_layers=2,
                                              num_decoder_layers=0
                                              ).encoder
        else:
            self.seg_encoder = nn.Transformer(d_model=encoder.config.hidden_size,
                                              nhead=encoder.config.num_attention_heads,
                                              batch_first=True,
                                              dim_feedforward=dim_feedforward,
                                              activation=activation,
                                              dropout=dropout,
                                              layer_norm_eps=layer_norm_eps,
                                              num_encoder_layers=2,
                                              num_decoder_layers=0).encoder

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        # Hypothetical Example
        # Batch of 4 documents: (batch_size, n_segments, max_segment_length) --> (4, 64, 128)
        # BERT-BASE encoder: 768 hidden units

        # Squash samples and segments into a single axis (batch_size * n_segments, max_segment_length) --> (256, 128)
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))
        else:
            token_type_ids_reshape = None

        # Encode segments with BERT --> (256, 128, 768)

        if self.encoder.config.model_type in ['distilbert', 'mt5']:
            encoder_outputs = self.encoder(input_ids=input_ids_reshape,
                                           attention_mask=attention_mask_reshape)[0]
        else:
            encoder_outputs = self.encoder(input_ids=input_ids_reshape,
                                           attention_mask=attention_mask_reshape,
                                           token_type_ids=token_type_ids_reshape)[0]

        # Reshape back to (batch_size, n_segments, max_segment_length, output_size) --> (4, 64, 128, 768)
        encoder_outputs = encoder_outputs.contiguous().view(input_ids.size(0),
                                                            self.max_segments,
                                                            self.max_segment_length,
                                                            self.hidden_size)

        # Gather CLS outputs per segment --> (4, 64, 768)
        encoder_outputs = encoder_outputs[:, :, 0]

        # Infer real segments, i.e., mask paddings (4, 64)
        seg_mask = (torch.sum(input_ids, 2) != 0).to(input_ids.dtype)
        # Infer and collect segment positional embeddings (4, 64)
        seg_positions = torch.arange(1, self.max_segments + 1).to(input_ids.device) * seg_mask
        # Add segment positional embeddings to segment inputs (4, 64, 768)
        encoder_outputs += self.seg_pos_embeddings(seg_positions)

        # Encode segments with segment-wise transformer (4, 64, 768)
        seg_encoder_outputs = self.seg_encoder(encoder_outputs)

        # Collect document representation by aggregating segment representations (4, 768)
        outputs, _ = torch.max(seg_encoder_outputs, 1)

        return SimpleOutput(last_hidden_state=outputs, hidden_states=outputs)


def build_hierarchical_model(model, max_segments, max_segment_length):
    config = model.config
    # Hack the classifier encoder to use hierarchical BERT
    if config.model_type in supported_models:
        if config.model_type == 'bert':
            segment_encoder = model.bert
        elif config.model_type == 'distilbert':
            segment_encoder = model.distilbert
        elif config.model_type in ['roberta', 'xlm-roberta']:
            segment_encoder = model.roberta
        elif config.model_type == 'deberta-v2':
            segment_encoder = model.deberta
        elif config.model_type == 'camembert':
            segment_encoder = model.roberta
        elif config.model_type == 'mt5':
            segment_encoder = model.mt5
        # Replace flat BERT encoder with hierarchical BERT encoder
        model_encoder = HierarchicalBert(encoder=segment_encoder,
                                         max_segments=max_segments,
                                         max_segment_length=max_segment_length)
        if config.model_type == 'bert':
            model.bert = model_encoder
        elif config.model_type == 'distilbert':
            model.distilbert = model_encoder
        elif config.model_type in ['roberta', 'xlm-roberta']:
            model.roberta = model_encoder
        elif config.model_type == 'deberta-v2':
            model.deberta = model_encoder
        elif config.model_type == "camembert":
            model.roberta = model_encoder
        elif config.model_type == 'mt5':
            model.mt5 = model_encoder
    elif config.model_type in ['longformer', 'big_bird']:
        pass
    else:
        raise NotImplementedError(f"{config.model_type} is not supported yet!")

    return model


# Some model tokenizers are based on roberta (https://huggingface.co/docs/transformers/model_doc/roberta)
# a RoBERTa tokenizer, derived from the GPT-2 tokenizer, uses byte-level Byte-Pair-Encoding.
# This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
# be encoded differently whether it is at the beginning of the sentence (without space) or not
# You need to pass add_prefix_space=True to make it work, otherwise an error will occur
models_that_require_add_prefix_space = ["iarfmoose/roberta-base-bulgarian", "gerulata/slovakbert", "roberta-base",
                                        "PlanTL-GOB-ES/roberta-base-bne", "bertin-project/bertin-roberta-base-spanish",
                                        "BSC-TeMU/roberta-base-bne", "pdelobelle/robbert-v2-dutch-base",
                                        "roberta-large"]


def get_tokenizer(model_name_or_path, revision):
    # https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384: They explicitly state that "This checkpoint uses BertModel with XLMRobertaTokenizer so AutoTokenizer won't work with this checkpoint!".
    # However, after refactoring, using XLMRobertaTokenizer causes some errors: ValueError: word_ids() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast` class).
    # AutoTokenizer works anyway
    # AutTokenizer gives the following infos: This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods.
    tokenizer_class = AutoTokenizer
    if model_name_or_path in models_that_require_add_prefix_space:
        return tokenizer_class.from_pretrained(model_name_or_path, add_prefix_space=True, revision=revision)
    else:
        return tokenizer_class.from_pretrained(model_name_or_path, revision=revision)


def get_model_class_for_sequence_classification(model_type, model_args=None):
    model_type_to_model_class = {
        "distilbert": HierDistilBertForSequenceClassification,
        "deberta-v2": HierDebertaV2ForSequenceClassification,
        "roberta": HierRobertaForSequenceClassification,
        "xlm-roberta": HierXLMRobertaForSequenceClassification,
        "camembert": HierCamembertForSequenceClassification,
        "mt5": HierT5ForSequenceClassification
        # Here just get directly the encoder, because there is no transformers class MT5ForSequenceClassification
    }
    if model_args is not None:
        if model_type in model_type_to_model_class.keys() and model_args.hierarchical == True:
            return model_type_to_model_class[model_type]
        else:
            if model_type == 'mt5':
                return T5ForSequenceClassification
            else:
                return AutoModelForSequenceClassification
    else:
        return AutoModelForSequenceClassification


if __name__ == "__main__":
    from transformers import AutoModel

    model_names = [
        "bert-base-uncased",  # standard model
        "distilbert-base-multilingual-cased",  # special case with different class
        "microsoft/mdeberta-v3-base",  # special case with different class
        "roberta-base",  # special case with different class TODO somehow there is still a problem with roberta
        "xlm-roberta-base",  # special case with different class
        "microsoft/Multilingual-MiniLM-L12-H384",  # special case with different tokenizer
    ]

    for model_name in model_names:
        print("Testing model", model_name)
        tokenizer = get_tokenizer(model_name)

        # Use as a stand-alone encoder
        encoder = AutoModel.from_pretrained(model_name)
        num_examples = 4  # batch size
        max_segments = 8  # smaller for faster debugging
        max_segment_length = 32  # smaller for faster debugging
        model = HierarchicalBert(encoder=encoder, max_segments=max_segments, max_segment_length=max_segment_length)

        fake_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
        model_type = encoder.config.model_type
        for i in range(num_examples):
            # Tokenize segment
            temp_inputs = tokenizer(['dog ' * (max_segment_length - 2)] * max_segments, return_token_type_ids=True)
            fake_inputs['input_ids'].append(temp_inputs['input_ids'])
            fake_inputs['attention_mask'].append(temp_inputs['attention_mask'])
            if model_type != 'distilbert':  # distilbert has no token_type_ids
                fake_inputs['token_type_ids'].append(temp_inputs['token_type_ids'])

        fake_inputs['input_ids'] = torch.as_tensor(fake_inputs['input_ids'])
        fake_inputs['attention_mask'] = torch.as_tensor(fake_inputs['attention_mask'])
        if model_type != 'distilbert':
            fake_inputs['token_type_ids'] = torch.as_tensor(fake_inputs['token_type_ids'])
        else:
            fake_inputs['token_type_ids'] = None

        output = model(fake_inputs['input_ids'], fake_inputs['attention_mask'], fake_inputs['token_type_ids'])

        # 4 document representations of 768 or 384 features (hidden size) are expected
        assert output[0].shape == torch.Size([num_examples, 768]) or output[0].shape == torch.Size([num_examples, 384])

        # Use with HuggingFace AutoModelForSequenceClassification and Trainer API

        # Init Classifier
        num_labels = 10
        model_class = get_model_class_for_sequence_classification(model_type)
        model = model_class.from_pretrained(model_name, num_labels=num_labels)
        model = build_hierarchical_model(model, max_segments, max_segment_length, HierarchicalBert)

        output = model(fake_inputs['input_ids'], fake_inputs['attention_mask'], fake_inputs['token_type_ids'])

        # 4 document outputs with 10 (num_labels) logits are expected
        assert output.logits.shape == torch.Size([num_examples, num_labels])
