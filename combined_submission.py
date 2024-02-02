"""
Add description
"""

from sklearn.metrics import label_ranking_average_precision_score
from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.loader import DataLoader, PrefetchLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model5 import Model
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
from info_nce import InfoNCE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers.optimization import get_linear_schedule_with_warmup

from sklearn.metrics.pairwise import cosine_similarity


CE = torch.nn.CrossEntropyLoss()
INCE = InfoNCE() # Contrastive Predictive Coding; van den Oord, et al. 2018
def contrastive_loss(v1, v2, beta=0.1):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return beta * (CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)) + (1 - beta) * (INCE(v1, v2) + INCE(v2, v1))

class customContrastiveLoss(torch.nn.Module):
    """ Contrastive loss function.
    The main difference between this and the standard contrastive loss is that this one uses the InfoNCE loss instead of the cross entropy loss. 
    We also keep in memory last seen batch of text and graph embeddings to compute the loss.
    """
    def __init__(self, memory = 5):
        super(customContrastiveLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.ince = InfoNCE()
        self.memory = memory
        if self.memory > 0:
            self.memory_text = None
            self.memory_graph = None
    def reset_memeory(self):
        self.memory_text = None
        self.memory_graph = None
    def forward(self, v1, v2):
        logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
        labels = torch.arange(logits.shape[0], device=v1.device)
        result = self.ince(v1, v2, self.memory_graph) + self.ince(v2, v1, self.memory_text)
        if self.memory > 0:
            with torch.no_grad():
                if self.memory_text is None:
                    self.memory_text = v1.detach()
                    self.memory_graph = v2.detach()
                else:
                    self.memory_text = torch.cat((self.memory_text, v1), 0)
                    self.memory_graph = torch.cat((self.memory_graph, v2), 0)
                    if self.memory_text.shape[0] > self.memory:
                        self.memory_text = self.memory_text[-self.memory:]
                        self.memory_graph = self.memory_graph[-self.memory:]
        return result


save_directory = "./combined_dataset"
model_name = 'allenai/scibert_scivocab_uncased'

tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("../data/token_embedding_dict.npy", allow_pickle=True)[()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = 100
batch_size = 80
learning_rate = 1e-4
bert_lr = 3e-5

model = Model(model_name=model_name, num_node_features=300, nout=1024, nhid=1024, graph_hidden_channels=1024, conv_type="GAT", nheads=8, nlayers=3)  # nout is changed to 2048
model.to(device)
parameter_string = 'model_name: {}, num_node_features: {}, nout: {}, nhid: {}, graph_hidden_channels: {}, conv_type: {}, nheads: {}'.format(model_name, 300, 2048, 500, 500, "GAT", 10)  # Adjusted nout to 2048
print(parameter_string)
print(batch_size)
print(learning_rate)
best_validation_score = 0

optimizer = optim.AdamW([{'params': model.parameters_no_bert},
                         {'params': model.text_encoder.parameters(), 'lr': bert_lr}], lr=learning_rate,
                        betas=(0.9, 0.999),
                        weight_decay=0.01)

linear_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=nb_epochs-20)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000
custom_loss = customContrastiveLoss( memory=batch_size*2)
print(optimizer.param_groups[0]['lr'])


if os.path.exists(save_directory):
    for file in os.listdir(save_directory):
        if file.startswith('model'):
            save_path = os.path.join(save_directory, file)
print('loading best model...')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root='../data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='../data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size//2, shuffle=False)

graph_embeddings = []
for batch in test_loader:
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size//2, shuffle=False)
text_embeddings = []
for batch in test_text_loader:
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())



similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv(save_directory +'/submission.csv', index=False)