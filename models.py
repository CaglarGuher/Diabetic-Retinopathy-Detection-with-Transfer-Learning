
import torchvision.models as models
from torch import nn


model_dict = {
    "densenet161": {"model": models.densenet161, "weight": 0.6},
    "resnet152": {"model": models.resnet152, "weight": 0.55},
    "resnet101": {"model": models.resnet101, "weight": 0.5},
    "vgg19": {"model": models.vgg19, "weight": 0.4},
    "alexnet": {"model": models.alexnet, "weight": 0.25},
    "googlenet": {"model": models.googlenet, "weight": 0.45},
    "mobilenet_v2": {"model": models.mobilenet_v2, "weight": 0.35},
    "shufflenet_v2_x1_0": {"model": models.shufflenet_v2_x1_0, "weight": 0.3},
    "resnext50_32x4d": {"model": models.resnext50_32x4d, "weight": 0.7},
    "resnext101_32x8d": {"model": models.resnext101_32x8d, "weight": 0.8},
    "resnext101_64x4d": {"model": models.resnext101_64x4d, "weight": 0.85},
    "wide_resnet50_2": {"model": models.wide_resnet50_2, "weight": 0.65},
    "wide_resnet101_2": {"model": models.wide_resnet101_2, "weight": 0.75},
    "efficient-netb7": {"model": models.efficientnet_b7, "weight": 1.0},
    "efficient-netb6": {"model": models.efficientnet_b6, "weight": 0.95},
    "efficient-netb5": {"model": models.efficientnet_b5, "weight": 0.9},
}


def select_model(model_name):
    
   

    
 
    if model_name in model_dict:
        model_func = model_dict[model_name]["model"]
        if  model_name == "resnet152":
            model = model_func(weights = "ResNet152_Weights.DEFAULT" )
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        ) 
            

        elif  model_name == "resnet101":
            model = model_func(weights = "ResNet101_Weights.DEFAULT" )
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        ) 
        elif  model_name == 'googlenet':
            model = model_func(weights = "GoogLeNet_Weights.DEFAULT" )
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )
        elif  model_name == "shufflenet_v2_x1_0":
            model = model_func(weights = "ShuffleNet_V2_X1_0_Weights.DEFAULT" )
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )
        elif  model_name == "resnext50_32x4d":
            model = model_func(weights = "ResNeXt50_32X4D_Weights.DEFAULT" )
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )
        elif  model_name == "resnext101_32x8d":
            model = model_func(weights = "ResNeXt101_32X8D_Weights.DEFAULT" )
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )
        elif  model_name == "resnext101_64x4d":
            model = model_func(weights = "ResNeXt101_64X4D_Weights.DEFAULT" )
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )
        
        elif  model_name == "resnext50_32x4d":
            model = model_func(weights = "ResNeXt50_32X4D_Weights.DEFAULT" )
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )
        elif  model_name == "wide_resnet50_2":
            model = model_func(weights = "Wide_ResNet50_2_Weights.DEFAULT")
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )
        elif  model_name == "wide_resnet101_2":
            model = model_func(weights = "Wide_ResNet101_2_Weights.DEFAULT")
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )
             
                                 
        elif model_name == "vgg19":
            model = model_func(weights = "VGG19_Weights.IMAGENET1K_V1")
            for parameter in model.parameters():
                parameter.requires_grad = False
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                        nn.Linear(num_ftrs, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1))
            
        elif model_name == "alexnet":
            model = model_func(weights = "AlexNet_Weights.DEFAULT")
            for parameter in model.parameters():
                parameter.requires_grad = False
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                        nn.Linear(num_ftrs, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1))

        elif model_name=="efficient-netb7":
            model = model_func(weights = "EfficientNet_B7_Weights.DEFAULT")
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.classifier[1] =nn.Sequential(
                        nn.Linear(model.classifier[1].in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )

        elif model_name=="efficient-netb6":
            model = model_func(weights = "EfficientNet_B6_Weights.DEFAULT")
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.classifier[1] =nn.Sequential(
                        nn.Linear(model.classifier[1].in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )

        elif model_name=="efficient-netb5":
            model = model_func(weights = "EfficientNet_B5_Weights.DEFAULT")
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.classifier[1] =nn.Sequential(
                        nn.Linear(model.classifier[1].in_features, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        ) 
            
        
        elif model_name == "mobilenet_v2":
            model = model_func(weights = "MobileNet_V2_Weights.DEFAULT")
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.classifier[1] =  nn.Sequential(
                        nn.Linear(model.last_channel, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )
        elif model_name == "densenet161":
            model = model_func(weights = "DenseNet161_Weights.DEFAULT")
            for parameter in model.parameters():
                parameter.requires_grad = False
            num_ftrs = model.classifier.in_features
                
            model.classifier = nn.Sequential(
                        nn.Linear(num_ftrs, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )
    return model