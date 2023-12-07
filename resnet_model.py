# resnet_model.py
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

def ResnetCNN(num_classes): 
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model