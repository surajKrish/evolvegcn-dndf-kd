import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric_temporal.nn.recurrent import EvolveGCNO


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes, dropout_rate=0.5):
        super(RecurrentGCN, self).__init__()
        self.dropout_rate = dropout_rate
        self.recurrent_1 = EvolveGCNO(node_features, num_classes)
        self.recurrent_2 = EvolveGCNO(node_features, num_classes)
        self.linear = torch.nn.Linear(node_features, num_classes)

    def forward(self, data):
        x = self.recurrent_1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.recurrent_2(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)

    def embed(self, data):
        x = self.recurrent_1(data.x, data.edge_index)
        return x
