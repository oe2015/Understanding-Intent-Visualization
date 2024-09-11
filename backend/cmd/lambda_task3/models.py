from torch import nn
from transformers import XLMRobertaModel


class PersuasionModel(XLMRobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(config.hidden_size, 24))

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
