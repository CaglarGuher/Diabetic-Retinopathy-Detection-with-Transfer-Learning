
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
        "resnext101_32x8d":models.resnext101_32x8d,
        "resnext101_64x4d":models.resnext101_64x4d,
        "wide_resnet50_2": models.wide_resnet50_2,
        "wide_resnet101_2":models.wide_resnet101_2,
        "efficient-netb7":models.efficientnet_b7,
        "efficient-netb6":models.efficientnet_b6,
        "efficient-netb5":models.efficientnet_b5,
    }
    
    # Return the specified model
    if model_name in model_dict:
        model_func = model_dict[model_name]
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
        if  model_name == 'googlenet':
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
        if  model_name == "shufflenet_v2_x1_0":
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
        if  model_name == "resnext50_32x4d":
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
        if  model_name == "resnext101_32x8d":
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
        if  model_name == "resnext101_64x4d":
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
        
        if  model_name == "resnext50_32x4d":
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
        if  model_name == "wide_resnet50_2":
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
        if  model_name == "wide_resnet101_2":
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
            print(num_ftrs)
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
            print(num_ftrs)
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
            model.classifier[1] =   model.classifier[1] =nn.Sequential(
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
            model.classifier[1] =   model.classifier[1] =nn.Sequential(
                        nn.Linear(model.last_channel, 256),  
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, 5),
                        nn.Softmax(dim=1)
                        )
    return model