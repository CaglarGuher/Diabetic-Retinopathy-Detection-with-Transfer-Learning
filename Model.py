import torchvision.models as models
import torch

def select_model(model_name):
    
    # Define dictionary of model names and their corresponding models
    model_dict = {
        "resnet152": models.resnet152,
        "resnet101":models.resnet101,
        "vgg19": models.vgg19,
        "densenet161": models.densenet161,
        "inception_v3": models.inception_v3,
        "alexnet": models.alexnet,
        "googlenet": models.googlenet,
        "mobilenet_v2": models.mobilenet_v2,
        "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
        "resnext50_32x4d": models.resnext50_32x4d,
        "wide_resnet50_2": models.wide_resnet50_2,
        "efficient-netb7":models.efficientnet_b7,
        "efficient-netb6":models.efficientnet_b6,
        "efficient,netb5":models.efficientnet_b5,
    }
    
    # Return the specified model
    if model_name in model_dict:
        model_func = model_dict[model_name]
        model = model_func(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features, 5)
        return model
    
    # Return None if the specified model is not found
    else:
        return None