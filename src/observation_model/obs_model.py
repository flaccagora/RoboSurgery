import torch
import torch.nn as nn
import torch.nn.functional as F

class singleNN(nn.Module):
    def __init__(self):
        super(singleNN, self).__init__()

        self.f1 = nn.Linear(6, 128)
        self.f2 = nn.Linear(128, 128)
        self.f3 = nn.Linear(128, 128)
        self.f4 = nn.Linear(128,1)
        
    def forward(self, pos,theta,dim=1):
        x = torch.cat([pos,theta], dim=dim)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.sigmoid(self.f4(x))
        return x
    
class cardinalNN(nn.Module):
    def __init__(self):
        super(cardinalNN, self).__init__()

        self.f1 = nn.Linear(6, 128)
        self.f2 = nn.Linear(128, 128)
        self.f3 = nn.Linear(128, 128)
        self.f4 = nn.Linear(128,4)
        
    def forward(self, pos,theta,dim=1):
        x = torch.cat([pos,theta], dim=dim)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.sigmoid(self.f4(x))
        return x
    
