
import torchvision.models as models
from torch import nn

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
        "efficient-netb5":models.efficientnet_b5,
    }
    
    # Return the specified model
    if model_name in model_dict:
        model_func = model_dict[model_name]
        if any(name in model_name for name in ['resnet','googlenet','shufflenet','resnext','wide_resnet']):
            model = model_func(pretrained=True)
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )  
        elif any(name in model_name for name in ['inception']):
            model = model_func(weights='IMAGENET1K_V1')
            model.aux_logits = False
            for parameter in model.parameters():
                parameter.requires_grad = False
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,5)
                                 
        elif any(name in model_name for name in ["vgg","alex"]):
            model = model_func(pretrained=True)
            for parameter in model.parameters():
                parameter.requires_grad = False
            num_ftrs = model.classifier[6].in_features
            print(num_ftrs)
            model.classifier[6] = nn.Sequential(
                        nn.Linear(num_ftrs, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1))

        elif any(name in model_name for name in["efficient"]):
            model = model_func(pretrained=True)
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.classifier[1] =nn.Sequential(
                        nn.Linear(model.classifier[1].in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        ) 
        elif any(name in model_name for name in["mobilenet"]):
            model = model_func(pretrained=True)
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.classifier[1] =   model.classifier[1] =nn.Sequential(
                        nn.Linear(model.last_channel, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )
        elif any(name in model_name for name in["densenet"]):
            model = model_func(pretrained=True)
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.classifier  =nn.Sequential(
                        nn.Linear(model.classifier.in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )  


    return model