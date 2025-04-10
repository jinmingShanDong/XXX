import torch
import torch.nn as nn

class HashFeatureNetwork(nn.Module):
    def __init__(self, hash_bit):
        super(HashFeatureNetwork, self).__init__()
        self.fc1 = nn.Linear(hash_bit, hash_bit * 2)  
        self.drop = nn.Dropout(0.3)               
        self.relu1 = nn.ReLU()          
        
        self.fc2 = nn.Linear(hash_bit * 2, hash_bit * 2)  
        self.fc3 = nn.Linear(hash_bit * 2, hash_bit)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

