import sys
import torch
import torch.nn as nn
import numpy as np
import os
from config.base_config import Config
from model.second_level_space import contrastive_loss
from model.first_level_space import VideoFeatureTransformer 
from model.three_level_space import HashFeatureNetwork
from torch.nn.functional import normalize
import torch.nn.functional as F


class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface: 
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        self.dp = nn.Dropout(0.3)
        self.activation1_video = nn.ReLU()
        self.fc1_video = nn.Linear(self.config.embed_dim, self.config.hash_bit)
        self.last_layer_video = nn.Tanh()
        self.hash_layer_video = nn.Sequential(self.activation1_video, self.dp, self.fc1_video, self.last_layer_video)

        self.activation1_text = nn.ReLU() 
        self.fc1_text = nn.Linear(self.config.embed_dim, self.config.hash_bit)
        self.last_layer_text = nn.Tanh()  
        self.hash_layer_text = nn.Sequential(self.activation1_text, self.dp, self.fc1_text, self.last_layer_text)
        
       
        self.t = nn.Parameter(((torch.rand(config.batch_size, config.hash_bit, requires_grad=True) * 2) - 1))
        self.t_L = nn.Linear(self.config.hash_bit, self.config.hash_bit)

        self.zero_level_semantic_space = VideoFeatureTransformer(dim=self.config.embed_dim, num_heads=8, num_layers=6)
  
        self.hash_bit = self.config.hash_bit
        self.fu_linear = nn.Linear(2 * self.config.hash_bit, self.config.hash_bit)
        self.second_level_space = HashFeatureNetwork(self.config.hash_bit)
        self.cosine_similarity_loss = contrastive_loss

    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        if self.config.huggingface: 
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
        else:
            text_features = self.clip.encode_text(text_data) 
            video_features = self.clip.encode_image(video_data) 
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1) 
 
        if return_all_frames == False: # train
            text_video_features_list = [text_features.unsqueeze(1), video_features] 
            text_video_features = torch.cat(text_video_features_list, dim=1) 
            text_video_features = self.zero_level_semantic_space(text_video_features) 
            shared_text_features = text_video_features[:,0,:] 
            shared_video_features = text_video_features[:,1:,:] 
            text_features = shared_text_features
            video_features = shared_video_features

        if return_all_frames == True: 
            text_features  = self.zero_level_semantic_space(text_features.unsqueeze(1)).squeeze(1) 
            video_features = self.zero_level_semantic_space(video_features)
        video_features = torch.mean(video_features, dim=1, keepdim=True) 
        video_features = video_features.squeeze(1) 

        text_features_hash_3 = self.dp(text_features)
        video_features_hash_3 = self.dp(video_features)

        if return_all_frames == False:
            text_video_features_hash_list = [text_features_hash_3, video_features_hash_3]
            text_video_features_hash = self.fu_linear(torch.cat(text_video_features_hash_list, dim=1)) 
            text_video_features_hash = self.second_level_space(text_video_features_hash)  
            loss_level1 = self.cosine_similarity_loss(text_features_hash_3, video_features_hash_3) + (self.cosine_similarity_loss(text_features_hash_3, text_video_features_hash) + self.cosine_similarity_loss(video_features_hash_3, text_video_features_hash))
        if return_all_frames == True: 
            text_text_features_hash_list = [text_features_hash_3, text_features_hash_3]
            text_features_hash_3 = self.fu_linear(torch.cat(text_text_features_hash_list, dim=1)) 
            text_features_hash_3 = self.second_level_space(text_features_hash_3) 
      
            video_video_features_hash_list = [video_features_hash_3, video_features_hash_3]
            video_features_hash_3 = self.fu_linear(torch.cat(video_video_features_hash_list, dim=1))
            video_features_hash_3 = self.second_level_space(video_features_hash_3) 

        text_features_hash = text_features_hash_3
        video_features_hash = video_features_hash_3

        tv = self.t_L(self.t.cuda()) 

        mid_text_video_features_hash = (text_features_hash + video_features_hash) / torch.tensor(2.0)

        loss_a1_b1 = self.custom_loss(text_features_hash, video_features_hash, mid_text_video_features_hash, tv)

        if return_all_frames==False:
            total_loss =  loss_level1 + loss_a1_b1
        if return_all_frames== True:
            total_loss = loss_a1_b1
        final_text_features_hash = torch.where(text_features_hash > tv, torch.tensor(1.0).cuda(), torch.tensor(-1.0).cuda()) # [32, hash_bit]
        final_video_features_hash = torch.where(video_features_hash > tv, torch.tensor(1.0).cuda(), torch.tensor(-1.0).cuda()) # [32, hash_bit]
        
        return text_features, video_features, final_text_features_hash, final_video_features_hash, total_loss
    
    def cosine_similarity_loss(self, text_features, video_features, margin=0.1):
        cos_sim = F.cosine_similarity(text_features, video_features, dim=1)
        loss = 1 - cos_sim
        loss = torch.clamp(loss - margin, min=0)
        return loss.mean()
    
    def custom_loss(self, text_features_hash, video_features_hash, mid_text_video_features_hash, tv):
        text_mid_distance = torch.norm(text_features_hash - mid_text_video_features_hash, dim=1)
        
        video_mid_distance = torch.norm(video_features_hash - mid_text_video_features_hash, dim=1)

        text_video_distance = torch.norm(text_features_hash-video_features_hash)
        
        close_loss = text_mid_distance.mean() + video_mid_distance.mean() + text_video_distance.mean()

        mid_tv_distance = torch.norm(mid_text_video_features_hash - tv, dim=1)
        far_loss = 1 / (mid_tv_distance.mean() + 1e-8)  
        total_loss = close_loss + far_loss
        return total_loss

