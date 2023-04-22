# Diabetic Retinopathy Detection using Deep Learning CNN Architectures

This project aims to develop a deep learning-based system for the detection of Diabetic Retinopathy (DR) with 5 stages which are

*  level 1:No diabetic retinopathy.
*  level 2: Mild nonproliferative diabetic retinopathy.
*  level 3: Moderate nonproliferative diabetic retinopathy. 
*  level 4: Severe nonproliferative diabetic retinopathy. 
*  level 5: Proliferative diabetic retinopathy.


by employing various Convolutional Neural Networks (CNN) architectures with over 90000 eye fundus images. Diabetic Retinopathy is a diabetes-related complication that affects the retina, which is the light-sensitive layer at the back of the eye. If left untreated, DR can lead to severe vision loss or even blindness. Early detection and intervention are crucial for mitigating the risk of vision loss in patients with diabetes.

The project utilizes 16 different CNN models to classify images of the retina into different stages of DR, ranging from no DR to severe DR. The employed models are as follows:

1. Densenet161
2. Resnet152
3. Resnet101
4. VGG19
5. AlexNet
6. GoogleNet
7. MobileNet V2
8. ShuffleNet V2 x1.0
9. ResNeXt50 32x4d
10. ResNeXt101 32x8d
11. ResNeXt101 64x4d
12. Wide ResNet50_2
13. Wide ResNet101_2
14. EfficientNet-B7
15. EfficientNet-B6
16. EfficientNet-B5

These models are fine-tuned using transfer learning and hyperparameter optimization to achieve the best possible performance on the task of DR detection.

The training process involves several steps, including pre-processing of the retinal images, splitting the dataset into training and validation sets, and hyperparameter tuning for each model. The dataset used for this project consists of thousands of high-resolution retinal images sourced from a reliable repository. The images are first pre-processed to ensure uniformity and optimal input for the CNN models.

Once the models are trained, a weighted ensemble method is employed to combine the predictions of all 16 models. This approach aims to leverage the strengths of each model and improve the overall performance of the system in detecting DR. The models are assigned weights based on their individual performance, and a weighted voting scheme is used to determine the final prediction for each retinal image.

The ultimate goal of this project is to develop an accurate and efficient system for DR detection that can be integrated into clinical workflows and aid ophthalmologists in identifying DR in its early stages. By automating the detection process and reducing the reliance on manual examination, this system has the potential to significantly improve patient outcomes and reduce the burden on healthcare professionals.

Please note that this project is still under development, with the training phase of the models remaining to be completed. Stay tuned for updates and improvements to the system.