import torch
from torch import nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from utils import split_title_content


class FramingModel(XLMRobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(768 * 2, 14))
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("./models/xlm-roberta-base")

    def forward(self, input_text):
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
        title_input_ids = title_tok["input_ids"]
        title_attention_mask = title_tok["attention_mask"]
        content_input_ids = content_tok["input_ids"]
        content_attention_mask = content_tok["attention_mask"]
        output_title = super().forward(
            input_ids=title_input_ids, attention_mask=title_attention_mask
        )[0][:, 0, :]
        output_content = super().forward(
            input_ids=content_input_ids, attention_mask=content_attention_mask
        )[0][:, 0, :]
        x = torch.cat((output_title, output_content), 1)
        classes = self.classifier(x)
        return classes
