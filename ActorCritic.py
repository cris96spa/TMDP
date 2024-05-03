import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNet(nn.Module):
    def __init__(self, nS, nA, hidden_dim=16):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.device = device
        self.embedding = nn.Embedding(nS, nS) # Embedding layer to handle integer states
        self.hidden = nn.Linear(nS, hidden_dim)
        self.output = nn.Linear(hidden_dim, nA)

    def forward(self, s):
        # Embed the integer state
        s = self.embedding(s)
        s = F.relu(self.hidden(s))
        logits = self.output(s)
        return logits
    
    def get_logits(self):
        full_logits = np.zeros((self.nS, self.nA))
        with torch.no_grad():
            for s in range(self.nS):
                s = torch.tensor([s], dtype=torch.long).to(self.device)
                logits = self.forward(s)
                logits = logits.squeeze(dim=0)
                full_logits[s] = logits.cpu().numpy()
        return full_logits

    def act(self, s):
        with torch.no_grad():
            # Preparing the input state tensor
            s = torch.tensor([s], dtype=torch.long).to(self.device)  # Wrap integer state in a list, convert to tensor
            logits = self.forward(s)
            logits = logits.squeeze(dim=0)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1)
            return action.item()
        
    def act_and_log_prob(self, s):
        with torch.no_grad():
            # Preparing the input state tensor
            s = torch.tensor([s], dtype=torch.long).to(self.device)
            logits = self.forward(s)
            logits = logits.squeeze(dim=0)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).squeeze(dim=0)
            logprb = -F.cross_entropy(logits, action, reduction='none')
            return action.tolist(), logits.tolist(), logprb.tolist()

    def get_probabilities(self):
        probs = np.zeros((self.nS, self.nA))
        with torch.no_grad():
            for s in range(self.nS):
                s = torch.tensor([s], dtype=torch.long).to(self.device)
                logits = self.forward(s)
                logits = logits.squeeze(dim=0)
                probs[s] = F.softmax(logits, dim=-1).cpu().numpy()
        return probs
    
    

class ValueNet(nn.Module):
    def __init__(self, nS, hidden_dim=16):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(nS, nS) # Embedding layer to handle integer states
        self.hidden = nn.Linear(nS, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        # Embed the integer state
        s = self.embedding(s)
        s = F.relu(self.hidden(s))
        logits = self.output(s)
        return logits
    
    def get_value(self, s):
        with torch.no_grad():
            s = torch.tensor([s], dtype=torch.long).to(self.device)
            value = self.forward(s).cpu().numpy()
        return value
    
    def get_values(self):
        values = np.zeros(self.nS)
        with torch.no_grad():
            for s in range(self.nS):
                s = torch.tensor([s], dtype=torch.long).to(self.device)
                values[s] = self.forward(s).cpu().numpy()
        return values
    
class QNet(nn.Module):
    def __init__(self, nS, nA, hidden_dim=16):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.device = device
        self.embedding_s = nn.Embedding(nS, nS) # Embedding layer to handle integer states
        self.embedding_a = nn.Embedding(nA, nA)
        self.hidden = nn.Linear(nS+nA, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        # Embed the integer state
        s = self.embedding_s(s)
        a = self.embedding_a(a)
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.hidden(x))
        logits = self.output(x)
        return logits
    
    def get_value(self, s, a):
        with torch.no_grad():
            s = torch.tensor([s], dtype=torch.long).to(self.device)
            a = torch.tensor([a], dtype=torch.long).to(self.device)
            value = self.forward(s, a).cpu().numpy()
        return value

    def get_values(self):
        values = np.zeros((self.nS, self.nA))
        with torch.no_grad():
            for s in range(self.nS):
                for a in range(self.nA):
                    s = torch.tensor([s], dtype=torch.long).to(self.device)
                    a = torch.tensor([a], dtype=torch.long).to(self.device)
                    values[s, a] = self.forward(s, a).cpu().numpy()
        return values