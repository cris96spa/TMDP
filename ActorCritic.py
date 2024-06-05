import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
class ActorNet(nn.Module):
    def __init__(self, nS, nA, hidden_dim=64):
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
    
    def sample(self, s):
        logits = self.forward(s).squeeze(dim=0)
        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, num_samples=1)
        log_prob = torch.log(probs).gather(1, actions)
        return actions, log_prob
    
    def get_probs(self, s):
        logits = self.forward(s).squeeze(dim=0)
        probs = F.softmax(logits, dim=-1)
        return probs
        
    def get_probabilities(self):
        probs = np.zeros((self.nS, self.nA))
        with torch.no_grad():
            for s in range(self.nS):
                s = torch.tensor([s], dtype=torch.long).to(self.device)
                logits = self.forward(s)
                logits = logits.squeeze(dim=0)
                probs[s] = F.softmax(logits, dim=-1).cpu().numpy()
        return probs
    
    # Soft update of the actor network
    def soft_update(self, reference, alpha):
        for param, ref_param in zip(self.parameters(), reference.parameters()):
            param.data.copy_(alpha * ref_param.data + (1 - alpha) * param.data)

class ValueNet(nn.Module):
    def __init__(self, nS, hidden_dim=64):
        super().__init__()
        self.nS = nS
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

    def soft_update(self, reference, alpha):
        for param, ref_param in zip(self.parameters(), reference.parameters()):
            param.data.copy_(alpha * ref_param.data + (1 - alpha) * param.data)


class QNet(nn.Module):
    def __init__(self, nS, nA, hidden_dim=64):
        super().__init__()
        self.nS = nS
        self.nA = nA
        self.device = device
        self.embedding_s = nn.Embedding(nS, nS) # Embedding layer to handle integer states
        self.hidden = nn.Linear(nS, hidden_dim)
        self.output = nn.Linear(hidden_dim, nA)

    def forward(self, s):
        # Embed the integer state
        x = self.embedding_s(s)
        x = F.relu(self.hidden(x))
        logits = self.output(x)
        return logits
    
    def get_value(self, s):
        with torch.no_grad():
            s = torch.tensor([s], dtype=torch.long).to(self.device)
            value = self.forward(s).cpu().numpy()
        return value

    def get_values(self):
        values = np.zeros((self.nS, self.nA))
        with torch.no_grad():
            for s in range(self.nS):
                s = torch.tensor([s], dtype=torch.long).to(self.device)
                values[s] = self.forward(s).cpu().numpy()
        return values
   
    def soft_update(self, reference, alpha):
        for param, ref_param in zip(self.parameters(), reference.parameters()):
            param.data.copy_(alpha * ref_param.data + (1 - alpha) * param.data)
