import pandas as pd
import numpy as np 
import cv2
import PIL.Image as Image 
import torch
from torch.utils.data import Dataset,DataLoader 
from sklearn.model_selection import train_test_split
from torchvision import transforms






def get_data(data_label,train_test_path,val_path,train_test_sample_size,batch_size,image_filter,model):

    class dataset(Dataset): # Inherits from the Dataset class.

        def __init__(self,df,data_path,image_filter = None,image_transform=None): # Constructor.
            super(Dataset,self).__init__() #Calls the constructor of the Dataset class.
            self.df = df
            self.data_path = data_path
            self.image_transform = image_transform
            self.image_filter = image_filter
            self.model = model

            
        def __len__(self):
            return len(self.df) #Returns the number of samples in the dataset.
        
        def __getitem__(self,index):
            image_id = self.df['image'][index]
            image = cv2.imread(f'{self.data_path}/{image_id}.jpg') #Image.

            resize_224 = transforms.Compose([transforms.Resize([224,224])])
            resize_229 = transforms.Compose([transforms.Resize([229,229])])
            resize_600 = transforms.Compose([transforms.Resize([600,600])])
            resize_528 = transforms.Compose([transforms.Resize([528,528])])
            resize_456 = transforms.Compose([transforms.Resize([456,456])])
            
            if self.image_filter:
                image = self.image_filter(image)

            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if self.image_transform :


                if self.model in ["resnet152", "resnet101", "vgg19", "densenet161", "alexnet", "googlenet", 
                                  "mobilenet_v2", "shufflenet_v2_x1_0", "resnext50_32x4d", "wide_resnet50_2"]:
                    
                    image = resize_224(image)

                elif self.model == "inception_v3":
                    image = resize_229(image)

                elif self.model == "efficient-netb7":
                    image = resize_600(image)
                
                elif self.model == "efficient-netb6":
                    image = resize_528(image)

                elif self.model == "efficient,netb5":
                    image = resize_456(image)


                image = self.image_transform(image) #Applies transformation to the image.
                
            
            label = self.df['level'][index] #Label.
        
            return image,torch.tensor(label) #If train == True, return image & label.

        
            

    df_test_train = (data_label[data_label["validation"] == 0].sample(n = train_test_sample_size))
    max_count = df_test_train["level"].value_counts().max()
    balanced_dfs = []


    # For each unique class label
    for label in df_test_train["level"].unique():
        # Get the subset of samples with the current label
        subset = df_test_train[df_test_train["level"] == label]
        
        # Oversample the subset to match the maximum count
        oversampled_subset = subset.sample(n=max_count, replace=True)
        
        # Append the oversampled subset to the balanced dataset list
        balanced_dfs.append(oversampled_subset)
        
    # Concatenate the balanced datasets into a single dataframe
    balanced_train_test = pd.concat(balanced_dfs).reset_index()

   

    train_test_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-10, 10)),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=[-10, 10]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    data = dataset(balanced_train_test,f'{train_test_path}',image_transform = train_test_transform,image_filter=image_filter)




    train_set,valid_set = train_test_split(data,test_size=0.2,random_state=42)

    df_validation = data_label[data_label["validation"] ==1]

    df_validation= df_validation.reset_index()

    

    valid_data = dataset(df_validation,f'{val_path}',image_transform = valid_transform,image_filter=image_filter)


    train_dataloader = DataLoader(train_set,batch_size=batch_size,shuffle=True) #DataLoader for train_set.
    test_dataloader = DataLoader(valid_set,batch_size=batch_size,shuffle=False) #DataLoader for test_set.
    valid_dataloader = DataLoader(valid_data,batch_size =batch_size,shuffle=False) #validate model with 2500 eye images (500 for each class)

    return train_dataloader,test_dataloader,valid_dataloader