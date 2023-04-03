import os
import pandas as pd
import cv2 
import numpy as np
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
print(desktop_path)
dataset_total = pd.read_csv("C:/Users/PC/Desktop/retinopathy_data/data/labels/label.csv")
train_test_image_directory = "C:/Users/PC/Desktop/retinopathy_data/data/test_train_images"
validation_directiory = "C:/Users/PC/Desktop/retinopathy_data/data/validation_images"


def extract_bv(image):
    b,green_fundus,r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)

    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)	
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
    contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    #vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)	
    xmask = np.ones(fundus_eroded.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)   
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"	
        else:
            shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
    blood_vessels = cv2.bitwise_not(finimage)
    return blood_vessels

#FOR TAGGING
def blurry_or_not(laplacian_value):
    
    if laplacian_value <10000:
        return 0
    else:
        return 1
  




import os
label_for_level_1 = dataset_total[dataset_total['level'] == 1]
label_for_level_2 = dataset_total[dataset_total['level'] == 2]
label_for_level_0 = dataset_total[dataset_total['level'] == 0]
label_for_level_3 = dataset_total[dataset_total['level'] == 3]
label_for_level_4 = dataset_total[dataset_total['level'] == 4]
import os
source_dir = "C:/Users/PC/Desktop/retinopathy_data/data/test_train_images"
destination_dir = "C:/Users/PC/Desktop/retinopathy_data/data/validation_images"

# Loop through the DataFrame and move images to the destination directory based on the level
for i, row in label_for_level_0[-10000:].iterrows():
    image_name = row['image']
    level = row['level']
 
    src_path = f"{source_dir}/{image_name}.jpg"
    dst_path = f"{destination_dir}/{image_name}.jpg"
    os.rename(src_path, dst_path)


# Loop through the DataFrame and move images to the destination directory based on the level
for i, row in label_for_level_1[-2500:].iterrows():
    image_name = row['image']
    level = row['level']
 
    src_path = f"{source_dir}/{image_name}.jpg"
    dst_path = f"{destination_dir}/{image_name}.jpg"
    os.rename(src_path, dst_path)

# Loop through the DataFrame and move images to the destination directory based on the level
for i, row in label_for_level_2[-1250:].iterrows():
    image_name = row['image']
    level = row['level']
 
    src_path = f"{source_dir}/{image_name}.jpg"
    dst_path = f"{destination_dir}/{image_name}.jpg"
    os.rename(src_path, dst_path)

# Loop through the DataFrame and move images to the destination directory based on the level
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
       
print(total)