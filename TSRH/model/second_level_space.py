import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SemanticSpace(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SemanticSpace, self).__init__()
        self.text_projection = nn.Linear(input_dim, output_dim)
        self.video_projection = nn.Linear(input_dim, output_dim)

    def forward(self, text_features, video_features):
        text_embedding = self.text_projection(text_features)
        video_embedding = self.video_projection(video_features)
        return text_embedding, video_embedding

def contrastive_loss(text_embedding, video_embedding, margin=1.0):
    cos_sim = F.cosine_similarity(text_embedding, video_embedding, dim=-1)
    labels = torch.ones_like(cos_sim)  
    positive_loss = (1 - cos_sim).mean() 
    negative_loss = F.relu(cos_sim - margin).mean()  # 如果距离小于 margin，就增加损失
    
    loss = positive_loss + negative_loss
    return loss

