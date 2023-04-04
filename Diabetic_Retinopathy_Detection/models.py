from webbrowser import get
import torchvision.models as models
import torch

def get_model(model_name):
    """given a model name, returns a pretrained and frozen model"""
    if(model_name=="inception_v3"):
        model = models.inception_v3(pretrained=True)
        model.aux_logits = False
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features, 8)
    elif(model_name=="resnet50"):
        model = models.resnet18(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features, 8)
    elif(model_name=="alexnet"):
        model = models.alexnet(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 8)
    elif(model_name=="vgg"):
        model = models.vgg11_bn(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 8)
    elif(model_name=="squeezenet"):
        model = models.squeezenet1_0(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.classifier[1] = torch.nn.Conv2d(512, 8, kernel_size=(1,1), stride=(1,1))
    elif(model_name=="densenet"):
        model = models.densenet121(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features, 8)
    elif(model_name=="efficientnet_b0"):
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 8)
        for param in model.features.parameters():
            param.requires_grad = False
    elif(model_name=="convnext_tiny"):
        model = models.convnext_tiny(pretrained=True)
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 8)   
    return model
    
