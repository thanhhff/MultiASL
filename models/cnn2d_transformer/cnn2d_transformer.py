import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn2d_transformer.video_extraction.resnet_2d import ResNetFeatureExtractor
from models.cnn2d_transformer.temporal_transformer.transformer import TemporalTransformer


class CNN2D_Transformer(nn.Module):
    def __init__(self, video_encoder, len_feature, num_classes, num_segments, fusion_type, config=None):
        super(CNN2D_Transformer, self).__init__()
        # Fusion
        self.fusion_type = fusion_type

        # Spatial Encoder
        self.video_encoder = ResNetFeatureExtractor(video_encoder, image_pretrained=True, image_trainable=True)

        # Temporal Transformer
        self.temporal_transformer = TemporalTransformer(dim=len_feature, depth=1, heads=4, dim_head=128, dropout=0.1, scale_dim=4)

        if self.fusion_type == 'concat':
            self.mlp_head = nn.Linear(len_feature * 4, num_classes) # 4 devices
        else:
            self.mlp_head = nn.Linear(len_feature, num_classes)

        self.base_module = nn.Conv1d(in_channels=len_feature * 2, out_channels=num_classes, kernel_size=1, padding=0)
        self.action_module_rgb = nn.Conv1d(in_channels=len_feature, out_channels=1, kernel_size=1, padding=0)
        self.action_module_flow = nn.Conv1d(in_channels=len_feature, out_channels=1, kernel_size=1, padding=0)
        
    def forward(self, x, is_training, reweight_cas=False, do_supp=True, kmult=6, pool=False):
        """
        Input:
            x: (batch_size, num_devices, num_segments, D, H, W)
        """
        batch_size, num_devices, num_segments, D, H, W = x.size()

        ### Spatial Encoder -> Features
        x_video_feature = x.view(-1, x.size(3), x.size(4), x.size(5))
        x_video_feature = self.video_encoder(x_video_feature)
        x_video_feature = x_video_feature.view(batch_size, num_devices, num_segments, -1)

        ### Temporal Transformer
        x_temporal_transformer = x_video_feature.view(-1, num_segments, x_video_feature.size(3))
        x_cls_transformer, x_temporal_transformer = self.temporal_transformer(x_temporal_transformer)
        x_temporal_transformer = x_temporal_transformer.view(batch_size, num_devices, num_segments, -1)

        ### Aggregate Features by Max on Devices Axis
        if self.fusion_type == 'max':
            x_cls_aggregate = x_cls_transformer.view(batch_size, num_devices, -1).max(dim=1)[0]
        elif self.fusion_type == 'mean':
            x_cls_aggregate = torch.mean(x_cls_transformer.view(batch_size, num_devices, -1), dim=1)
        elif self.fusion_type == 'sum':
            x_cls_aggregate = torch.sum(x_cls_transformer.view(batch_size, num_devices, -1), dim=1)
        elif self.fusion_type == 'concat':
            x_cls_aggregate = x_cls_transformer.reshape(batch_size, -1)

        x_rgb = x_video_feature.max(dim=1)[0]
        x_flow = x_temporal_transformer.max(dim=1)[0]

        ### Concat RGB and Flow
        x_rgb_flow = torch.cat((x_rgb, x_flow), dim=2)

        cas_fg = self.base_module(x_rgb_flow.permute(0, 2, 1)).permute(0, 2, 1)
        action_rgb = torch.sigmoid(self.action_module_rgb(x_rgb.permute(0, 2, 1)))
        action_flow = torch.sigmoid(self.action_module_flow(x_flow.permute(0, 2, 1)))
        
        ### Video Classification
        x_cls = self.mlp_head(x_cls_aggregate)

        ### Add softmax
        # x_cls = F.log_softmax(x_cls, dim=1)

        return x_cls, cas_fg, action_flow, action_rgb
