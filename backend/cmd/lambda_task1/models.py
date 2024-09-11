from torch import nn
from transformers import AutoConfig, AutoModel


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained("./models/xlm-roberta-base")
        self.model = AutoModel.from_config(config)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 3)  # Assuming output size of 3 as in your original class

    def forward(self, inputs):
        outputs = self.model(**inputs)
        embeddings = outputs[1]  # Assuming index 1 is the pooler_output
        embeddings = self.dropout(embeddings)
        output = self.fc(embeddings)
        return output
