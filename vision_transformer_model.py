import torch.nn as nn
import timm

def VisionTransformerModel(num_classes):

    model = timm.create_model('mobilevit_xs.cvnets_in1k', pretrained=True)
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    return model