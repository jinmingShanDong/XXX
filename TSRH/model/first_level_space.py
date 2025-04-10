import torch
import torch.nn as nn
class VideoFeatureTransformer(nn.Module):
    def __init__(self, dim, num_heads, num_layers, dropout=0.1):
        super(VideoFeatureTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, video_features_1):
        video_features_1 = video_features_1.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(video_features_1)
        transformer_output = transformer_output.permute(1, 0, 2)
        return transformer_output
