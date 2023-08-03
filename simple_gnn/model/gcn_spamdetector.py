import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCNSpamDetector(torch.nn.Module):
    def __init__(self, bert_model, tokenizer, hidden_channels, num_classes):
        super(GCNSpamDetector, self).__init__()
        self.bert = bert_model
        self.conv1 = GCNConv(bert_model.config.hidden_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.tokenizer = tokenizer

    def forward(self, data):
        inputs = {'input_ids': data.x.long(), 'attention_mask': (data.x != self.tokenizer.pad_token_id).long()}
        outputs = self.bert(**inputs)
        x = outputs.last_hidden_state
        x = self.conv1(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


