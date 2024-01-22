from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model3 import Model
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
from info_nce import InfoNCE
from torch.optim.lr_scheduler import ReduceLROnPlateau
#import warnings
#warnings.simplefilter("ignore", UserWarning)


CE = torch.nn.CrossEntropyLoss()
INCE = InfoNCE() # Contrastive Predictive Coding; van den Oord, et al. 2018
def contrastive_loss(v1, v2, beta=0.8):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return beta * (CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)) + (1 - beta) * (INCE(v1, v2) + INCE(v2, v1))

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = 100
batch_size = 64
learning_rate = 5e-5
bert_lr = 5e-6

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Model(model_name=model_name, num_node_features=300, nout=1024, nhid=300, graph_hidden_channels=1024, conv_type="GAT", nheads=10)  # nout is changed to 1024
model.to(device)
parameter_string = 'model_name: {}, num_node_features: {}, nout: {}, nhid: {}, graph_hidden_channels: {}, conv_type: {}, nheads: {}'.format(model_name, 300, 1024, 500, 500, "GAT", 10)  # Adjusted nout to 1024
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
    for batch in train_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch

        x_graph, x_text = model(graph_batch.to(device),
                                input_ids.to(device),
                                attention_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()

        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery), flush=True)
            losses.append(loss)
            loss = 0
    model.eval()
    val_loss = 0
    for batch in val_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        x_graph, x_text = model(graph_batch.to(device),
                                input_ids.to(device),
                                attention_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)
        val_loss += current_loss.item()
    lr_scheduler.step(val_loss)

    best_validation_loss = min(best_validation_loss, val_loss)
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) ,flush=True)
    if best_validation_loss==val_loss:
        print('validation loss improoved saving checkpoint...')
        save_path = os.path.join('./models/', 'model'+str(i)+'.pt')
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


from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv('submission.csv', index=False)