import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn3d import resnet_3d


class CNN3D(nn.Module):
    def __init__(self, num_classes, shortcut_type, sample_size, sample_duration, model_depth, mode='score'):
        super(CNN3D, self).__init__()

        if mode == 'score':
            self.last_fc = True
        elif mode == 'feature':
            self.last_fc = False

        if model_depth == 10:
            self.model = resnet_3d.resnet10(num_classes=num_classes, shortcut_type=shortcut_type,
                                    sample_size=sample_size, sample_duration=sample_duration,
                                    last_fc=self.last_fc)
            self.fc = nn.Linear(512, num_classes)
        elif model_depth == 18: 
            self.model = resnet_3d.resnet18(num_classes=num_classes, shortcut_type=shortcut_type,
                                    sample_size=sample_size, sample_duration=sample_duration,
                                    last_fc=self.last_fc)
            self.fc = nn.Linear(512, num_classes)
            
    def forward(self, x, is_training):
        """
        Input: x has shape [batch_size, num_device, sample_durarion, C, H, W]
        """
        # Convert x to shape [batch_size*num_device, sample_durarion, C, H, W] # [0, 1, 2, 3, 4]
        batch_size = x.size(0)
        num_devices = x.size(1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4), x.size(5))

        # Permute x to [batch_size*num_device, C, sample_durarion, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x) 
        x = self.model.maxpool(x)
        
        #### LAYER 1 ####
        x = self.model.layer1(x)
        #### LAYER 2 ####
        x = self.model.layer2(x)
        #### LAYER 3 ####
        x = self.model.layer3(x)
        #### LAYER 4 ####
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        # Convert x to shape [batch_size, num_device, sample_durarion, num_classes]
        x = x.view(batch_size, num_devices, -1)

        # Merge by max
        max_values, _ = torch.max(x, dim=1)
        x = self.fc(max_values)

        ### Add softmax
        x = F.log_softmax(x, dim=1)

        return x