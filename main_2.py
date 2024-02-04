"""
This script trains a model for graph-text classification and generates predictions.

Pipeline Overview:
1. Initialize necessary modules and variables.
2. Data Loading: Load graph and text datasets for training, validation, and testing.
3. Model Definition: Define the Graph-Text Fusion model architecture.
4. Define contrastive loss function (combining InfoNCE and CE).
5. Optimizer and Scheduler Setup: Setup AdamW optimizer with separate learning rates and linear scheduler with warm-up steps
6. Training Loop: Iterate over epochs and batches to train the model.
7. Validation: Evaluate the model's performance on a validation set / Save the model if validation loss improves.
8. Testing and Submission: Generate embeddings for the test dataset using the best model and create a CSV file with similarity scores.

Disclaimer : Some parts of this code were provided by the challenge organizers
"""


from sklearn.metrics import label_ranking_average_precision_score
from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.loader import DataLoader, PrefetchLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model1 import Model
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
from info_nce import InfoNCE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss import contrastive_loss
from sklearn.metrics.pairwise import cosine_similarity



save_directory = "./batches2"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
# model_name = 'distilbert-base-uncased'
model_name = 'allenai/scibert_scivocab_uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("../data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='../data/', gt=gt, split='val_scibert', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='../data/', gt=gt, split='train_scibert', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = 100
batch_size = 64
learning_rate = 5e-5
bert_lr = 5e-6

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_loader = PrefetchLoader(train_loader, device=device)
val_loader = PrefetchLoader(val_loader, device=device)

model = Model(model_name=model_name, num_node_features=300, nout=2048, nhid=300, graph_hidden_channels=2048, conv_type="GAT", nheads=8)  # nout is changed to 2048
model.to(device)
parameter_string = 'model_name: {}, num_node_features: {}, nout: {}, nhid: {}, graph_hidden_channels: {}, conv_type: {}, nheads: {}'.format(model_name, 300, 2048, 500, 500, "GAT", 10)  # Adjusted nout to 2048
print(parameter_string)
print(batch_size)
print(learning_rate)

optimizer = optim.AdamW([{'params': model.graph_encoder.parameters()},
                         {'params': model.text_encoder.parameters(), 'lr': bert_lr}], lr=learning_rate,
                        betas=(0.9, 0.999),
                        weight_decay=0.01)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True, min_lr=1e-7)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000



for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    model.train()
    for idx, batch in enumerate(train_loader):
        # print(idx)
        optimizer.zero_grad()
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch

        x_graph, x_text = model(graph_batch.to(device),
                                input_ids.to(device),
                                attention_mask.to(device))
        
        # accumulating 2 batches before backprop
        current_loss = contrastive_loss(x_text, x_graph)
        current_loss.backward()
        optimizer.step()
        current_loss = current_loss.detach().cpu()
        loss += current_loss.item()
        

        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery), flush=True)
            losses.append(loss)
            loss = 0
    optimizer.zero_grad()
    del graph_batch
    del input_ids
    del attention_mask
    del batch
    del current_loss
    
    model.eval()
    val_loss = 0
    val_text = []
    val_graph = []  
    for batch in val_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        x_graph, x_text = model(graph_batch.to(device),
                                input_ids.to(device),
                                attention_mask.to(device))
        val_text.append(x_text.tolist())
        val_graph.append(x_graph.tolist())
        current_loss = contrastive_loss(x_graph, x_text)
        x_graph = x_graph.detach()
        x_text = x_text.detach()
        current_loss = current_loss.detach()
        val_loss += current_loss.item()
    lr_scheduler.step(val_loss)
    val_text = np.concatenate(val_text)
    val_graph = np.concatenate(val_graph)
    similarity = cosine_similarity(val_text, val_graph)
    print('validation lrap: ', label_ranking_average_precision_score(np.eye(similarity.shape[0]), similarity), flush=True)

    best_validation_loss = min(best_validation_loss, val_loss)
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) ,flush=True)
    if best_validation_loss==val_loss:
        print('validation loss improoved saving checkpoint...')
        if os.path.exists(save_directory):
            for file in os.listdir(save_directory):
                if file.startswith('model'):
                    os.remove(os.path.join(save_directory, file))
        save_path = os.path.join(save_directory, 'model'+str(i)+'.pt')
        torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_accuracy': val_loss,
        'loss': loss,
        }, save_path)
        print('checkpoint saved to: {}'.format(save_path))


print('loading best model...')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

graph_embeddings = []
for batch in test_loader:
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
text_embeddings = []
for batch in test_text_loader:
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())



similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv('submission.csv', index=False)