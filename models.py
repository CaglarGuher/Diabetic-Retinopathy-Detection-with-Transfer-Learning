import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Your model dictionary
model_dict = {
    "ResNet152": {"model": models.resnet152, "weight": 1},
    "ResNeXt50_32X4D": {"model": models.resnext50_32x4d, "weight": 1.45},
    "ResNeXt101_32X8D": {"model": models.resnext101_32x8d, "weight": 2},
    "ResNeXt101_64X4D": {"model": models.resnext101_64x4d, "weight": 2.1},
    "Wide_ResNet50_2": {"model": models.wide_resnet50_2, "weight": 1.2},
    "Wide_ResNet101_2": {"model": models.wide_resnet101_2, "weight": 1.65},
    "EfficientNet_B7": {"model": models.efficientnet_b7, "weight": 2.4},
    "EfficientNet_B6": {"model": models.efficientnet_b6, "weight": 2.3},
    "EfficientNet_B5": {"model": models.efficientnet_b5, "weight": 2.2},
}

def modify_model(model, num_classes=5):
    if hasattr(model, 'fc'):
        print("thisone")
        in_features = model.fc.in_features
        last_layer = nn.Sequential(
            nn.BatchNorm1d(num_features=in_features),    
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        model.fc = last_layer
    elif hasattr(model, 'classifier'):
        in_features = model.classifier[1].in_features
        last_layer = nn.Sequential(
            nn.BatchNorm1d(num_features=in_features),    
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        model.classifier[1] = last_layer
    else:
        raise ValueError("Unsupported model type. Modify the function accordingly.")

    return model

def select_model(model_name, num_classes=5):
    if model_name in model_dict:
        model_func = model_dict[model_name]["model"]
        weight_key = f"{model_name}_Weights.DEFAULT"
        model = model_func(weights=weight_key)

        for parameter in model.parameters():
            parameter.requires_grad = False

        model = modify_model(model, num_classes)

    return model
