"""
This script implements fine-tuning for a pre-trained Graph-Text Fusion model on a another part of the dataset. 
Fine-tuning allows the model to adapt to specific characteristics of the new dataset while leveraging knowledge learned 
from the pre-training phase.

Pipeline Overview:

- Data Loading: Load the new dataset for fine-tuning using custom data loaders tailored for the combined dataset.
- Pre-Trained Model Loading: Load the pre-trained Graph-Text Fusion model, initializing it with weights learned during the pre-training phase.
- Optimizer and Scheduler Setup: Setup AdamW optimizer with separate learning rates and ReduceLROnPlateau scheduler for fine-tuning.
- Fine-Tuning Loop: Iterate over epochs and batches to fine-tune the model parameters while keeping the pre-trained weights frozen or partially frozen.
- Validation: Evaluate the fine-tuned model's performance on a validation set using cosine similarity.
- Testing and Submission: Generate embeddings for the test dataset using the fine-tuned model and create a submission file with similarity scores.

Note: 
- This code is an extension of the main code.
- It can be used once we have a pretrained model that we want to refine
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
    def __init__(self, beta=0.1, memory = 5):
        super(customContrastiveLoss, self).__init__()
        self.beta = beta
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
        # logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
        # labels = torch.arange(logits.shape[0], device=v1.device)
        result =  self.ince(v1, v2, self.memory_graph) + self.ince(v2, v1, self.memory_text)
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
save_directory = "./combined_fine_tune"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
# model_name = 'distilbert-base-uncased'
model_name = 'allenai/scibert_scivocab_uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("../data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='../data/', gt=gt, split='val_scibert', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='../data/', gt=gt, split='train_scibert', tokenizer=tokenizer)
combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = 100
batch_size = 80
learning_rate = 1e-5
bert_lr = 1e-6

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
train_loader = PrefetchLoader(train_loader, device=device)
val_loader = PrefetchLoader(val_loader, device=device)
combined_loader = PrefetchLoader(combined_loader, device=device)
model = Model(model_name=model_name, num_node_features=300, nout=1024, nhid=1024, graph_hidden_channels=1024, conv_type="GAT", nheads=8, nlayers=3, dropout = 0.1)  # nout is changed to 2048
model.to(device)
parameter_string = 'model_name: {}, num_node_features: {}, nout: {}, nhid: {}, graph_hidden_channels: {}, conv_type: {}, nheads: {}'.format(model_name, 300, 2048, 500, 500, "GAT", 10)  # Adjusted nout to 2048
print(parameter_string)
print(batch_size)
print(learning_rate)

optimizer = optim.AdamW([{'params': model.graph_encoder.parameters()},
                         {'params': model.text_encoder.parameters(), 'lr': bert_lr}], lr=learning_rate,
                        betas=(0.9, 0.999),
                        weight_decay=0.05)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-7)

loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000
last_seen_model = None
if os.path.exists("./combined_dataset.old"):
    for file in os.listdir("./combined_dataset.old"):
        if file.startswith('model'):
            last_seen_model = os.path.join("./combined_dataset.old", file)
if last_seen_model is not None:
    checkpoint = torch.load(last_seen_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('loaded model from: {}'.format(last_seen_model))
    epoch = int(last_seen_model.split('/')[-1].split('.')[0].split('model')[-1])
    best_validation_loss = checkpoint['validation_accuracy']*(1 + 5e-2)
    print('best validation loss: ', best_validation_loss/len(val_loader))

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7, )
    linear_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=nb_epochs)
    best_validation_score = checkpoint.get('validation_score', 0.885)
else:
    epoch = 0
del checkpoint

n_memory_epochs = 2
custom_loss = customContrastiveLoss(beta=0, memory=batch_size*n_memory_epochs)

for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    if i%10 == 0:
        n_memory_epochs += 1
        custom_loss.memory = batch_size*n_memory_epochs
    model.train()
    custom_loss.reset_memeory()
    for idx, batch in enumerate(combined_loader):
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
        current_loss = custom_loss(x_text, x_graph)
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
    linear_scheduler.step()
    optimizer.zero_grad()
    del graph_batch
    del input_ids
    del attention_mask
    del batch
    del current_loss
    
    model.eval()
    # we sample 5000 from the combined_set to compute the validation loss
    val_loss = 0
    val_text = []
    val_graph = []
    with torch.no_grad():
        for idx, batch in enumerate(combined_loader):
            if idx==41:
                break
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            x_graph, x_text = model(graph_batch.to(device),
                                    input_ids.to(device),
                                    attention_mask.to(device))
            x_graph = x_graph.detach()
            x_text = x_text.detach()
            val_text.append(x_text.tolist())
            val_graph.append(x_graph.tolist())
            current_loss = contrastive_loss(x_graph, x_text)
            current_loss = current_loss.detach()
            val_loss += current_loss.item()
        val_text = np.concatenate(val_text)
        val_graph = np.concatenate(val_graph)
        similarity = cosine_similarity(val_text, val_graph)
        mrr = label_ranking_average_precision_score(np.eye(similarity.shape[0]), similarity)
        print('validation lrap: ', mrr, flush=True)
        best_validation_score = max(best_validation_score, mrr)



        print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) ,flush=True)
        if best_validation_score==mrr:
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
            "best_validation_score": best_validation_score,
            }, save_path)
            print('checkpoint saved to: {}'.format(save_path))


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
solution.to_csv(save_directory + '/submission.csv', index=False)