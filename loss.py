import torch
from info_nce import InfoNCE
CE = torch.nn.CrossEntropyLoss()
INCE = InfoNCE() # Contrastive Predictive Coding; van den Oord, et al. 2018

def contrastive_loss(v1, v2, beta=0.1):
    """Compute the contrastive loss using a combination of cross-entropy (CE) loss and InfoNCE loss weighted by beta."""
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return beta * (CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)) + (1 - beta) * (INCE(v1, v2) + INCE(v2, v1))


class customContrastiveLoss(torch.nn.Module):
    """ Contrastive loss function.
    This custom loss function extends the standard contrastive loss by incorporating the InfoNCE loss, which allows
    for additional negative keys beyond the off-diagonal of the batch : we keep in memory last seen batch of text and 
    graph embeddings and we utlize them as negative samples. 
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
        # Reset memory buffers
        self.memory_text = None
        self.memory_graph = None
    def forward(self, v1, v2):
        # Compute the custom contrastive loss
        logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
        labels = torch.arange(logits.shape[0], device=v1.device)
        result = self.ince(v1, v2, self.memory_graph) + self.ince(v2, v1, self.memory_text)
       
        # Update memory
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
