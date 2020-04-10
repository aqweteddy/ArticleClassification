import torch
from torch import nn
from transformers import ElectraModel

class ElectraCls(nn.Module):
    def __init__(self, model_dir, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.electra = ElectraModel.from_pretrained(model_dir).cuda()
        self.linear1 = nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(self.electra.config.hidden_size, num_labels)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, inp_ids, seg_ids, inp_masks, labels=None):
        x = self.electra(inp_ids, seg_ids, inp_masks)[0]
        x = x[:, 0]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.linear2(x)
        if labels is None:
            return logits
        
        # print(x.size(), logits.size(), labels.size())
        loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits