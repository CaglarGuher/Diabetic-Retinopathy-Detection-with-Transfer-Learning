import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import logging
from PIL import Image


class data_adjust(Dataset):

    def __init__(self, df, data_path, model, image_filter=None, image_transform=None):
        super().__init__()
        self.df = df
        self.data_path = data_path
        self.image_transform = image_transform
        self.image_filter = image_filter
        self.model = model

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_id = self.df['image'][index]
        image = self.load_image(image_id)

        if self.image_filter:
            image = self.image_filter(image)

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.image_transform:
            image = self.apply_model_specific_resize(image)
            image = self.image_transform(image)

        label = self.df['level'][index]
        return image, torch.tensor(label)

    def load_image(self, image_id):
        return cv2.imread(f'{self.data_path}/{image_id}.jpg')

    def apply_model_specific_resize(self, image):
        model_resize_map = {
            "inception_v3": transforms.Resize([229, 229]),
            "efficient-netb7": transforms.Resize([600, 600]),
            "efficient-netb6": transforms.Resize([528, 528]),
            "efficient-netb5": transforms.Resize([456, 456]),
            "resnext101_32x8d": transforms.Resize([256, 256]),
            "resnext101_64x4d": transforms.Resize([256, 256])
        }

        default_resize = transforms.Resize([224, 224])

        resize_transform = model_resize_map.get(self.model, default_resize)
        return resize_transform(image)







def get_data(data_label, train_test_path, val_path, train_test_sample_size, batch_size, image_filter, model, validation=False):
    logging.info("Starting data preparation")
    
    df_test_train, df_validation = preprocess_data(data_label, train_test_sample_size)

    balanced_train_test = balance_data(df_test_train)

    train_test_transform, valid_transform = setup_transforms()

    if validation:
        train_dataloader, test_dataloader = None, None
        valid_dataloader = prepare_validation_dataloader(df_validation, val_path, valid_transform, image_filter, model)
    else:
        train_dataloader, test_dataloader = prepare_train_test_dataloaders(balanced_train_test, train_test_path, train_test_transform, image_filter, model, batch_size)
        valid_dataloader = None

    logging.info("Data preparation completed")

    return train_dataloader, test_dataloader, valid_dataloader

def preprocess_data(data_label, train_test_sample_size):
    df_test_train = data_label[data_label["validation"] == 0].sample(n=train_test_sample_size)
    df_validation = data_label[data_label["validation"] == 1].reset_index()
    
    return df_test_train, df_validation

def balance_data(df_test_train):
    logging.info("Balancing the train-test dataset")

    max_count = df_test_train["level"].value_counts().max()
    balanced_dfs = []

    for label in df_test_train["level"].unique():
        subset = df_test_train[df_test_train["level"] == label]
        oversampled_subset = subset.sample(n=max_count, replace=True)
        balanced_dfs.append(oversampled_subset)

    balanced_train_test = pd.concat(balanced_dfs).reset_index()
    
    return balanced_train_test

def setup_transforms():
    logging.info("Setting up data transforms")

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

    return train_test_transform, valid_transform

def prepare_validation_dataloader(df_validation, val_path, valid_transform, image_filter, model):
    logging.info("Preparing Validation DataLoader")

    valid_data = data_adjust(df_validation, val_path, image_transform=valid_transform, image_filter=image_filter, model=model)
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)

    return valid_dataloader

def prepare_train_test_dataloaders(balanced_train_test, train_test_path, train_test_transform, image_filter, model, batch_size):
    logging.info("Preparing Train and Test DataLoaders")
    data_train_test = data_adjust(balanced_train_test, train_test_path, image_transform=train_test_transform, image_filter=image_filter, model=model)
    train_set, test_set = train_test_split(data_train_test, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader