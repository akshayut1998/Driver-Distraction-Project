import timm
import torch.nn as nn

def EfficientNetModel(num_classes):

    model = timm.create_model('efficientnet_b0', pretrained=True)
    model.classifier = nn.Linear(
    in_features=model.classifier.in_features ,out_features=10)
    return model