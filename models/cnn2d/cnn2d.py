import torch
import torch.nn as nn
from models.cnn2d.resnet_2d import ResNetFeatureExtractor


class CNN2D(nn.Module):
    def __init__(self, video_encoder, len_feature, num_classes, num_segments, pooling_type=""):
        super(CNN2D, self).__init__()
        # Spatial Encoder
        self.video_encoder = ResNetFeatureExtractor(video_encoder, image_pretrained=True, image_trainable=True)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(len_feature),
            nn.Linear(len_feature, num_classes)
        )
        
        self.base_module = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=num_classes, kernel_size=1, padding=0),
        )

        self.action_module_rgb = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )


    def forward(self, x, is_training):
        """
        Input:
            x: (batch_size, num_devices, num_segments, D, H, W)
        """
        batch_size, num_devices, num_segments, D, H, W = x.size()

        ### Spatial Encoder -> Features
        x_video_feature = x.view(-1, x.size(3), x.size(4), x.size(5))
        x_video_feature = self.video_encoder(x_video_feature)
        x_video_feature = x_video_feature.view(batch_size, num_devices, num_segments, -1)

        ### For ASL
        x_ASL, _ = torch.max(x_video_feature, dim=1)
        cas_fg = self.base_module(x_ASL.permute(0, 2, 1)).permute(0, 2, 1)
        action_rgb = torch.sigmoid(self.action_module_rgb(x_ASL.permute(0, 2, 1)))

        # Max dim=1
        x_video_feature, _ = torch.max(x_video_feature, dim=2)
        # Average dim = 1
        x_video_feature = torch.mean(x_video_feature, dim=1)
        # MLP Head
        x = self.mlp_head(x_video_feature)

        return x, cas_fg, action_rgb, action_rgb