"""
This script defines a Model for integrating a graph encoder and text encoder for a specific problem.
Two variations of the graph encoder (GatEncoder and GatEncoder2) are provided, each with its own architecture configuration.
The text encoder is based on a modified version of BERT.

Modules:
- GatEncoder: Graph Attention Network (GAT) encoder with a customizable architecture.
- GatEncoder2: Modified version of the GAT encoder with different hyperparameters.
- TextEncoder: Text encoder based on a modified version of BERT.
- Model: Integrates the graph and text encoders to create a joint architecture for training.

Note:
- This setting is adopted to generate embeddings of dimension 2048.
- Adjustments to the hyperparameters and model architecture can be made through the constructor of the Model class.
- Refer to the original papers of the models here : 
    - https://arxiv.org/pdf/1710.10903.pdf
    - https://arxiv.org/pdf/1810.04805.pdf
"""

from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv as GATConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel

class GatEncoder(nn.Module):
    """
    Graph Attention Network encoder
    """
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, nheads=4):
        super(GatEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GATConv(num_node_features, graph_hidden_channels, heads=nheads, dropout=0.2, concat=True)
        self.conv2 = GATConv(graph_hidden_channels, graph_hidden_channels, heads=nheads, dropout=0.2, concat=True)
        self.conv3 = GATConv(graph_hidden_channels, graph_hidden_channels, heads=nheads, dropout=0.2, concat=True)
        self.head_linear1 = nn.Linear(graph_hidden_channels * nheads, 2048)
        self.head_linear2 = nn.Linear(graph_hidden_channels * nheads, 2048)
        self.head_linear3 = nn.Linear(graph_hidden_channels * nheads, 2048)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, 2048)
        self.mol_hidden2 = nn.Linear(2048, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = self.head_linear1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.head_linear2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.head_linear3(x)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        x = self.ln(x)
        return x
    
class GatEncoder2(nn.Module):
    """
    Modified version of the GAT encoder module.
    """
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, nheads=4):
        super(GatEncoder2, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        assert graph_hidden_channels % nheads == 0
        self.conv1 = GATConv(num_node_features, graph_hidden_channels//nheads, heads=nheads, dropout=0.4, concat=True)
        self.conv2 = GATConv(graph_hidden_channels, graph_hidden_channels//nheads, heads=nheads, dropout=0.4, concat=True)
        self.conv3 = GATConv(graph_hidden_channels, graph_hidden_channels//nheads, heads=nheads, dropout=0.4, concat=True)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, 2048)
        self.mol_hidden2 = nn.Linear(2048, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        x = self.ln(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_name):
        """
        Modified version of  Text encoder module based on transformers.
        """
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 2048)  # Change input and output size to 1024
        self.ln = nn.LayerNorm((2048))

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        return self.ln(self.linear(encoded_text.last_hidden_state[:, 0, :]))

class Model(nn.Module):
    """
    Combines the graph and text encoders to create the final architecture for training.
    """
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, conv_type='GCN', nheads=4):
        super(Model, self).__init__()
        if conv_type == 'GCN':
            self.graph_encoder = GatEncoder(num_node_features, nout, nhid, graph_hidden_channels, nheads)  # Change to GatEncoder
        elif conv_type == 'GAT':
            self.graph_encoder = GatEncoder2(num_node_features, nout, nhid, graph_hidden_channels, nheads)
        self.text_encoder = TextEncoder(model_name)

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder
