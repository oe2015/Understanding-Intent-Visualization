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


class CustomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model = AutoModel.from_pretrained(cfg.MODEL_NAME)

        # self.pool = nn.Identity()

        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(768, cfg.NUM_CLASSES)

    def forward(self, inputs):

        outputs = self.model(**inputs)
        # last_hidden_states, pooler_output

        # output for Binary Classification
        embeddings = outputs[1]
        embeddings = self.dropout(embeddings)

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


class FramingModel(XLMRobertaModel):
    def __init__(self, config, num_classes):
        super().__init__(config)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(768 * 2, num_classes)
        )

    def forward(
        self,
        title_input_ids,
        title_attention_mask,
        content_input_ids,
        content_attention_mask,
    ):
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
    def __init__(self, config, num_classes):
        super().__init__(config)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(config.hidden_size, num_classes)
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

