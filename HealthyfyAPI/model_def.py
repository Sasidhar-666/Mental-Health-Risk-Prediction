
import torch
import torch.nn as nn
from transformers import DistilBertModel
# model_def.py

class DistilBertRegressor(nn.Module):
    def __init__(self, hidden_size=768):
        super(DistilBertRegressor, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.regressor = nn.Linear(hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.regressor(pooled_output)
        





# # to run : uvicorn main:app --reload