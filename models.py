
import torchvision.models as models
from torch import nn


model_dict = {
    "densenet161": {"model": models.densenet161, "weight": 1},
    "resnet152": {"model": models.resnet152, "weight": 0.6},
    "resnet101": {"model": models.resnet101, "weight": 0.5},
    "vgg19": {"model": models.vgg19, "weight": 0.4},
    "alexnet": {"model": models.alexnet, "weight": 0.25},
    "googlenet": {"model": models.googlenet, "weight": 0.45},
    "mobilenet_v2": {"model": models.mobilenet_v2, "weight": 0.35},
    "shufflenet_v2_x1_0": {"model": models.shufflenet_v2_x1_0, "weight": 0.3},
    "resnext50_32x4d": {"model": models.resnext50_32x4d, "weight": 1.45},
    "resnext101_32x8d": {"model": models.resnext101_32x8d, "weight": 2},
    "resnext101_64x4d": {"model": models.resnext101_64x4d, "weight": 2.1},
    "wide_resnet50_2": {"model": models.wide_resnet50_2, "weight": 1.2},
    "wide_resnet101_2": {"model": models.wide_resnet101_2, "weight": 1.65},
    "efficient-netb7": {"model": models.efficientnet_b7, "weight": 3.0},
    "efficient-netb6": {"model": models.efficientnet_b6, "weight": 2.8},
    "efficient-netb5": {"model": models.efficientnet_b5, "weight": 2.6},
}



def modify_model(model, num_classes=5):
    num_ftrs = model.fc.in_features if hasattr(model, 'fc') else model.classifier.in_features
    new_classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        nn.Softmax(dim=1)
    )

    if hasattr(model, 'fc'):
        model.fc = new_classifier
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = new_classifier
        else:
            model.classifier = new_classifier

    return model

def select_model(model_name):
    if model_name in model_dict:
        model_func = model_dict[model_name]["model"]
        weight_key = f"{model_name}_weights_default"
        model = model_func(weights=weight_key)

        for parameter in model.parameters():
            parameter.requires_grad = False

        model = modify_model(model)

    return model