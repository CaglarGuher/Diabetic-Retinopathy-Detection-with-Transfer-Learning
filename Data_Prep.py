
import cv2
import PIL.Image as Image 
import torch
from torch.utils.data import Dataset

from torchvision import transforms

class data_adjust(Dataset): # Inherits from the Dataset class.

        def __init__(self,df,model,data_path,image_filter = None,image_transform=None): # Constructor.
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
            resize_256 = transforms.Compose([transforms.Resize([256,256])])
            resize_600 = transforms.Compose([transforms.Resize([600,600])])
            resize_528 = transforms.Compose([transforms.Resize([528,528])])
            resize_456 = transforms.Compose([transforms.Resize([456,456])])
            
            if self.image_filter:
                image = self.image_filter(image)

            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if self.image_transform :


                if self.model in ["resnet152", "resnet101", "vgg19", "densenet161", "alexnet", "googlenet","wide_resnet101_2", 
                                  "mobilenet_v2", "shufflenet_v2_x1_0", "resnext50_32x4d", "wide_resnet50_2",]:
                    
                    image = resize_224(image)

                elif self.model == "inception_v3":
                    image = resize_229(image)

                elif self.model == "efficient-netb7":
                    image = resize_600(image)
                
                elif self.model == "efficient-netb6":
                    image = resize_528(image)

                elif self.model == "efficient-netb5":
                    image = resize_456(image)

                elif self.model in  ["resnext101_32x8d","resnext101_64x4d"]:
                    image = resize_256(image)
                    

                image = self.image_transform(image) #Applies transformation to the image.
                
            
            label = self.df['level'][index] #Label.
        
            return image,torch.tensor(label) #If train == True, return image & label.





