import torch.nn as nn
import torchvision

"""
Key design point:
Layer in each component and what layer needs to initialize?
"""


class backbone(nn.Module):
    def __init__(self, num_layer = 50):
        super(backbone, self).__init__()
        self.num_resnet_layers = num_layer
        self.backbone = None
        self.filter = []
        self.initialization_layer = []
        if self.num_resnet_layers == 18:
            self.backbone = torchvision.models.resnet18(pretrained=True)
            self.filters = [64, 64, 128, 256, 512]
        elif self.num_resnet_layers == 34:
            self.backbone = torchvision.models.resnet34(pretrained=True)
            self.filters = [64, 64, 128, 256, 512]
        elif self.num_resnet_layers == 50:
            self.backbone = torchvision.models.resnet50(pretrained=True)
            self.filters = [64, 256, 512, 1024, 2048]
        elif self.num_resnet_layers == 101:
            self.backbone = torchvision.models.resnet101(pretrained=True)
            self.filters = [64, 256, 512, 1024, 2048]
        elif self.num_resnet_layers == 152:
            self.backbone = torchvision.models.resnet152(pretrained=True)
            self.filters = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError('num_resnet_layers should be 18, 34, 50, 101 or 152')
    
    def forward(self):
        # No data involved at here and thus let it pass
        pass
    

