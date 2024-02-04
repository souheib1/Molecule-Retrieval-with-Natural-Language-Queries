"""
This script defines the Model we implemented for the problem combining a graph encoder and text encoder.
The graph encoder consists of a GAT (Graph Attention Network) encoder: GatEncoder.
The text encoder is based on BERT with modifications and includes an additional MLP layer.

Modules:
- GatEncoder: GAT encoder with customizable architecture including layer normalization and pooling options.
- MLP: Multi-layer perceptron with customizable activation function.
- TextEncoder: Modified BERT-based text encoder with an additional MLP layer.
- Model: Combines the graph and text encoders to create a joint architecture for training.

Note: 
- This setting is adopted to generate embeddings of dimension 1024.
- Adjustments to the hyperparameters and model architecture can be made through the constructor of the Model class.
- Refer to the original papers of the models here : 
    - https://arxiv.org/pdf/1710.10903.pdf
    - https://arxiv.org/pdf/1810.04805.pdf
    - https://arxiv.org/abs/1903.10676.pdf
"""


from torch import nn
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv as GATConv, GAT
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel

class GatEncoder(nn.Module):
    """
    GAT encoder with customizable architecture.    
    """
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, nheads = 4, nlayers = 3, layer_norm = True, pooling = "mean", dropout = 0.1):
        super(GatEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        if pooling == "mean":
            self.pooling = "mean"
            self.gat = GAT(num_node_features, graph_hidden_channels,nout=nout, heads=nheads, dropout=dropout, v2=True, num_layers = nlayers)
            self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
            self.mol_hidden2 = nn.Linear(nhid, nout)
        elif pooling is None:
            self.pooling = None
            assert graph_hidden_channels % nheads == 0
            graph_hidden_channels = graph_hidden_channels // nheads
            self.layers = nn.ModuleList()
            self.layers.append(GATConv(num_node_features, graph_hidden_channels, heads=nheads, dropout=0.1, concat=True))
            for i in range(nlayers - 1):
                self.layers.append(GATConv(graph_hidden_channels * nheads, graph_hidden_channels, heads=nheads, dropout=0.1, concat=True))


        self.layer_norm_flag = layer_norm

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        if self.pooling == "mean":
            x = self.gat(x, edge_index)
            x = global_mean_pool(x, batch)
            x = self.mol_hidden1(x).relu()
            x = self.mol_hidden2(x)
            if self.layer_norm_flag:
                x = self.ln(x)
        elif self.pooling is None:
            for layer in self.layers:
                x = layer(x, edge_index).relu()
        return x

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
    

class MLP(nn.Module):
    """
    Multi-layer perceptron with customizable activation function.    
    """
    def __init__(self, nin, nhid, nout, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.mol_hidden1 = nn.Linear(nin, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
        self.activation = activation
    def forward(self, x):
        x = self.mol_hidden1(x)
        x = self.activation(x)
        x = self.mol_hidden2(x)
        return x

class TextEncoder(nn.Module):
    """
    Modified version of the BERT-based text encoder with an additional MLP layer.    
    """
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 2048)  # Change input and output size to 1024
        self.MLP = MLP(768, 2048, 1024, torch.tanh)
        self.ln = nn.LayerNorm((1024))

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        return self.ln(self.MLP(encoded_text.last_hidden_state[:, 0, :]))

class Model(nn.Module):
    """
    Combines the graph and text encoders to create the final architecture for training.
    """
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, conv_type='GCN', nheads=4, temp=0.07, nlayers = 3, dropout = 0.1):
        super(Model, self).__init__()
        if conv_type == 'GCN':
            self.graph_encoder = GatEncoder(num_node_features, nout, nhid, graph_hidden_channels, nheads)  # Change to GatEncoder
        elif conv_type == 'GAT':
            self.graph_encoder = GatEncoder(num_node_features, nout, nhid, graph_hidden_channels, nheads, nlayers=nlayers, dropout = dropout)  # Change to GatEncoder
        self.temp = nn.Parameter(torch.Tensor([temp]))
        self.register_parameter( 'temp' , self.temp )
        self.parameters_no_bert = list(self.parameters())
        self.text_encoder = TextEncoder(model_name)

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        graph_encoded = graph_encoded*torch.exp(self.temp)
        text_encoded = text_encoded*torch.exp(self.temp)
        return graph_encoded, text_encoded

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder
