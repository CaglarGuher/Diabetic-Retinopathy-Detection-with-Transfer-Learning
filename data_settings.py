import os
import pandas as pd
import cv2
from utils import blurry_or_not, extract_bv

def main():

    dataset_total = pd.read_csv("C:/Users/PC/Desktop/retinopathy_data/data/labels/label.csv")
    train_test_image_directory = "C:/Users/PC/Desktop/retinopathy_data/data/test_train_images"
    validation_directory = "C:/Users/PC/Desktop/retinopathy_data/data/validation_images"

    separate_validation_set(dataset_total, train_test_image_directory, validation_directory)
    update_classification(dataset_total, train_test_image_directory)
    dataset_total.to_csv("C:/Users/PC/Desktop/retinopathy_data/data/labels/label.csv")

def separate_validation_set(dataset_total, source_dir, destination_dir):
    levels = range(5)
    samples_per_level = [10000, 2500, 1250, 450, 450]

    for level, num_samples in zip(levels, samples_per_level):
        label_for_level = dataset_total[dataset_total['level'] == level]

        for i, row in label_for_level[-num_samples:].iterrows():
            image_name = row['image']
            src_path = f"{source_dir}/{image_name}.jpg"
            dst_path = f"{destination_dir}/{image_name}.jpg"
            os.rename(src_path, dst_path)

    validation_images = [f.split('.')[0] for f in os.listdir(destination_dir)]
    dataset_total['validation'] = dataset_total['image'].apply(lambda x: 1 if x in validation_images else 0)

def update_classification(dataset_total, train_test_image_directory):
    for i in range(len(dataset_total)):
        if dataset_total["validation"][i] == 0 and dataset_total["classification"][i] == 0:
            try:
                image_path = f'{train_test_image_directory}/{dataset_total["image"][i]}.jpg'
                image = cv2.imread(image_path)
                image = cv2.resize(image, (512, 512))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                dataset_total["classification"][i] = blurry_or_not(cv2.Laplacian(extract_bv(image), cv2.CV_64F).var())

                if i % 1000 == 0:
                    print(f"%{i/len(dataset_total)} is finished")
            except:
                print(f"the image number of {i} has problem")

if __name__ == "__main__":
    main()