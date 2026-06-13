import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class CoarseToFineNet(nn.Module):
    def __init__(self, num_fine=11, num_coarse=4):
        super(CoarseToFineNet, self).__init__()
        # Load pre-trained EfficientNet-B0
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier[0] # The linear layer before output
        
        # Dual Heads
        self.fine_head = nn.Linear(1280, num_fine)
        self.coarse_head = nn.Linear(1280, num_coarse)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        fine_out = self.fine_head(x)
        coarse_out = self.coarse_head(x)
        return fine_out, coarse_out