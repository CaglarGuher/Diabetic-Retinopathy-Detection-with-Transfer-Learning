import os
import pandas as pd
import cv2 
import numpy as np
from utils import blurry_or_not , extract_bv
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

dataset_total = pd.read_csv("C:/Users/PC/Desktop/retinopathy_data/data/labels/label.csv")
train_test_image_directory = "C:/Users/PC/Desktop/retinopathy_data/data/test_train_images"
validation_directiory = "C:/Users/PC/Desktop/retinopathy_data/data/validation_images"


import os
label_for_level_1 = dataset_total[dataset_total['level'] == 1]
label_for_level_2 = dataset_total[dataset_total['level'] == 2]
label_for_level_0 = dataset_total[dataset_total['level'] == 0]
label_for_level_3 = dataset_total[dataset_total['level'] == 3]
label_for_level_4 = dataset_total[dataset_total['level'] == 4]
import os
source_dir = "C:/Users/PC/Desktop/retinopathy_data/data/test_train_images"
destination_dir = "C:/Users/PC/Desktop/retinopathy_data/data/validation_images"


for i, row in label_for_level_0[-10000:].iterrows():
    image_name = row['image']
    level = row['level']
 
    src_path = f"{source_dir}/{image_name}.jpg"
    dst_path = f"{destination_dir}/{image_name}.jpg"
    os.rename(src_path, dst_path)



for i, row in label_for_level_1[-2500:].iterrows():
    image_name = row['image']
    level = row['level']
 
    src_path = f"{source_dir}/{image_name}.jpg"
    dst_path = f"{destination_dir}/{image_name}.jpg"
    os.rename(src_path, dst_path)


for i, row in label_for_level_2[-1250:].iterrows():
    image_name = row['image']
    level = row['level']
 
    src_path = f"{source_dir}/{image_name}.jpg"
    dst_path = f"{destination_dir}/{image_name}.jpg"
    os.rename(src_path, dst_path)


for i, row in label_for_level_3[-450:].iterrows():
    image_name = row['image']
    level = row['level']
 
    src_path = f"{source_dir}/{image_name}.jpg"
    dst_path = f"{destination_dir}/{image_name}.jpg"
    os.rename(src_path, dst_path)
for i, row in label_for_level_4[-450:].iterrows():
    image_name = row['image']
    level = row['level']
 
    src_path = f"{source_dir}/{image_name}.jpg"
    dst_path = f"{destination_dir}/{image_name}.jpg"
    os.rename(src_path, dst_path)
validation_dir = "C:/Users/PC/Desktop/retinopathy_data/data/validation_images/"
import os
validation_images = [f.split('.')[0] for f in os.listdir(validation_dir)]
dataset_total['validation'] = dataset_total['image'].apply(lambda x: 1 if x in validation_images else 0)
  
dataset_total.to_csv("C:/Users/PC/Desktop/retinopathy_data/data/labels/label.csv")




total = []
for i in range(0,len(dataset_total)):
        if dataset_total["validation"][i] == 0 and dataset_total["classification"][i]==0:
        
                try:
                        number_of_defected = 0
                        image_path = f'{train_test_image_directory}{dataset_total["image"][i]}.jpg'
                        image = cv2.imread(image_path)
                        image = cv2.resize(image,(512,512))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        dataset_total["classification"][i] =  (blurry_or_not(cv2.Laplacian(extract_bv(image), cv2.CV_64F).var()))
                        if i % 1000 == 0 :
                                print(f"%{i/len(dataset_total)} is finished")

                except:
                        print(f"the image number of {i} has problem" )
       
