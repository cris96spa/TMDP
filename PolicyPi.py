import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyPi(nn.Module):
    def __init__(self, nS, nA, hidden_dim=64):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.device = device
        # Embedding layer to handle integer states
        self.embedding = nn.Embedding(nS, nS)
        self.hidden = nn.Linear(nS, hidden_dim)
        self.output = nn.Linear(hidden_dim, nA)

    def forward(self, s):
        # Embed the integer state
        s = self.embedding(s)
        s = F.relu(self.hidden(s))
        logits = self.output(s)
        return logits
    
    def act(self, s):
        with torch.no_grad():
            # Preparing the input state tensor
            s = torch.tensor([s], dtype=torch.long).to(self.device)  # Wrap integer state in a list, convert to tensor
            logits = self.forward(s)
            logits = logits.squeeze(dim=0)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1)
            return action.item()
    
    def get_probabilities(self):
        probs = np.zeros((self.nS, self.nA))
        with torch.no_grad():
            for s in range(self.nS):
                s = torch.tensor([s], dtype=torch.long).to(self.device)
                logits = self.forward(s)
                logits = logits.squeeze(dim=0)
                probs[s] = F.softmax(logits, dim=-1).cpu().numpy()
        return probs
    
    def get_logits(self):
        full_logits = np.zeros((self.nS, self.nA))
        with torch.no_grad():
            for s in range(self.nS):
                s = torch.tensor([s], dtype=torch.long).to(self.device)
                logits = self.forward(s)
                logits = logits.squeeze(dim=0)
                full_logits[s] = logits.cpu().numpy()
        return full_logits