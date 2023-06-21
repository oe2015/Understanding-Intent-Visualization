from logging import getLogger

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.xlm_roberta.configuration_xlm_roberta import (
    XLMRobertaOnnxConfig,
)
from transformers import XLMRobertaModel
from transformers import XLMRobertaTokenizer
from transformers.onnx.utils import (
    compute_effective_axis_dimension,
)
import torch
import torch.nn as nn

logger = getLogger(__name__)

# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# class CustomModel(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()

#         self.model = AutoModel.from_pretrained("xlm_roberta_base")

#         # self.pool = nn.Identity()

#         self.dropout = nn.Dropout(0.1)

#         self.fc = nn.Linear(768, 3)

#     def forward(self, inputs):

#         outputs = self.model(**inputs)
#         # last_hidden_states, pooler_output

#         # output for Binary Classification
#         embeddings = outputs[1]
#         embeddings = self.dropout(embeddings)

#         output = self.fc(embeddings)

#         return output
    

class CustomModel(XLMRobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(config.hidden_size, 3)  # you might want to adjust the output size

    def forward(self, input_ids, attention_mask):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # taking the pooler output (equivalent to `outputs.last_hidden_state[:, 0]`)
        embeddings = self.dropout(pooled_output)
        output = self.fc(embeddings)
        return output


def update_and_load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    for k in list(ckpt.keys()):
        if k.startswith("head2."):
            # rename the key to classifier
            ckpt[k.replace("head2.", "classifier.")] = ckpt.pop(k)
        if k.startswith("model."):
            # rename the key to classifier
            ckpt[k.replace("model.", "")] = ckpt.pop(k)

    return ckpt


def split_title_content(text):
    # Split the text by the first new line character
    parts = text.split('\n', 1)
    # Check if there is a new line character in the text
    if len(parts) > 1:
        title = parts[0]  # First part is the title
        content = parts[1]  # Second part is the content
    else:
        title = parts[0]  # Entire text is considered as the title
        content = ''
    # Split the title by the first period (.) to get the first sentence
    title_parts = title.split('.', 1)
    # Check if there is a period in the title
    if len(title_parts) > 1:
        first_sentence = title_parts[0] + '.'  # Add the period back to the first sentence
        title = title_parts[1]  # Remaining part becomes the new title
    return title.strip(), content.strip()


class FramingModel(XLMRobertaModel):
    def __init__(self, config):
        super().__init__(config)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(768 * 2, 14)
        )

        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    def forward(self, input_text):
        ###################################################################

        title, content = split_title_content(input_text)

        title_tok = self.tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            max_length=512,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        content_tok = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=512,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        title_input_ids = title_tok['input_ids']
        title_attention_mask = title_tok['attention_mask']
        content_input_ids = content_tok['input_ids']
        content_attention_mask = content_tok['attention_mask']

        ###################################################################
        output_title = super().forward(
            input_ids=title_input_ids, attention_mask=title_attention_mask
        )[0][:, 0, :]

        output_content = super().forward(
            input_ids=content_input_ids, attention_mask=content_attention_mask
        )[0][:, 0, :]

        x = torch.cat((output_title, output_content), 1)
        classes = self.classifier(x)

        return classes


class PersuasionModel(XLMRobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(config.hidden_size, 24)
        )

    def forward(
        self,
        content_input_ids,
        content_attention_mask,
    ):
        output_content = super().forward(
            input_ids=content_input_ids, attention_mask=content_attention_mask
        )[0][:, 0, :]

        classes = self.classifier(output_content)

        return classes

