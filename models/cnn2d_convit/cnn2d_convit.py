import timm
from torch import nn
import torch

class ConViT(nn.Module):
    def __init__(self, video_encoder='resnet18', pretrained=True, num_classes=10):
        super().__init__()
        self.backbone = timm.create_model(video_encoder, pretrained=pretrained, num_classes=768)
        net = timm.create_model('vit_base_resnet50_224_in21k', pretrained=True, num_classes=num_classes)
        self.transformer = net.blocks[0]
        self.transformer_head = net.head

        len_feature = 768
        self.base_module = nn.Conv1d(in_channels=len_feature * 2, out_channels=num_classes, kernel_size=1, padding=0)
        self.action_module_rgb = nn.Conv1d(in_channels=len_feature, out_channels=1, kernel_size=1, padding=0)
        self.action_module_flow = nn.Conv1d(in_channels=len_feature, out_channels=1, kernel_size=1, padding=0)

    def forward(self, images, is_training=True):
        """
        Args:
            images: (b, d, t, c, h, w) # batch, device, time, channel, height, width
        
        """
        b, d, t, c, h, w = images.size()
        images= images.reshape(-1, c, h, w)
        x = self.backbone(images)

        x_rgb = x.view(b, d, t, -1).max(dim=1)[0]

        x = x.reshape(b * d, t, -1)
        x = self.transformer(x)

        x_flow = x.view(b, d, t, -1).max(dim=1)[0]

        x = self.transformer_head(x)
        x = x[:, 0, :]
        x = x.reshape(b, d, -1)

        # Max
        # x = torch.max(x, dim=1)[0]

        # Average 
        # x = torch.mean(x, dim=1)

        # Sum
        x = torch.sum(x, dim=1)

        #### ASL
        x_rgb_flow = torch.cat((x_rgb, x_flow), dim=2)
        cas_fg = self.base_module(x_rgb_flow.permute(0, 2, 1)).permute(0, 2, 1)
        action_rgb = torch.sigmoid(self.action_module_rgb(x_rgb.permute(0, 2, 1)))
        action_flow = torch.sigmoid(self.action_module_flow(x_flow.permute(0, 2, 1)))

        return x, cas_fg, action_flow, action_rgb