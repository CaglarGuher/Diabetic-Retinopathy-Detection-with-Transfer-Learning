import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import PIL.Image as Imag 

def new_filter(img):

# Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a median filter to remove noise
    gray = cv2.medianBlur(gray, 5)

    # Apply adaptive thresholding to segment the blood vessels and little dots
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

    # Perform morphological opening to remove small objects
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return opening


def detect_symptoms(img):
    # Load the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply a median filter to remove noise
    gray = cv2.medianBlur(gray, 9)

    # Apply adaptive thresholding to segment the blood vessels and little dots
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Perform morphological opening to remove small objects
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Detect circles using Hough transform
    circles = cv2.HoughCircles(opening, cv2.HOUGH_GRADIENT, 1, 45, param1=30, param2=70, minRadius=25, maxRadius=75)

    # Detect hemorrhages using blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 45
    params.maxArea = 125
    params.filterByCircularity = True
    params.minCircularity = 0.7 
    params.maxCircularity = 1.0
    params.filterByConvexity = False
    params.filterByInertia = True
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    hemorrhages = []
    for kp in keypoints:
        x, y = np.int32(kp.pt)
        if opening[y,x] == 0:
            hemorrhages.append(kp)
            cv2.rectangle(img, (x-20, y-20), (x+20, y+20), (255,0,0), 2)

    # Detect exudates using morphological operatio

    # Display the results
    return img

def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image
def reduce_noise(image):
    # Apply median filtering to remove noise
    filtered = cv2.medianBlur(image, 5)

    return filtered

def crop_image(img,tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """  
    
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img>tol        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
def crop_image_only_outside(img,tol=0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]

def crop_eye_image(img):
    # Load the image


    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask of the eye
    _, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)

    # Find the contours of the eye in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (the outer boundary of the eye)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding box
    cropped_img = img[y:y+h, x:x+w]

    # Return the cropped image
    return cropped_img
def laptacian_filter(image):
    img_laplacian = cv2.Laplacian(image , cv2.CV_8U, ksize=5, scale=12)
    return img_laplacian
def cany_edge(img):
    canny = cv2.Canny(img, 50, 50)
    return canny
def blateral_filter(img):
    img_bilateral = cv2.bilateralFilter(img, 10, 5, 5)
    return img_bilateral
def sobel_filter(img):
# Apply the Sobel filter in the x direction
    sobel_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)

    # Apply the Sobel filter in the y direction
    sobel_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)

    # Combine the Sobel images using bitwise OR
    sobel = cv2.bitwise_or(sobel_x, sobel_y)

    return sobel
def sobel_filter(img):
# Apply the Sobel filter in the x direction
    sobel_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)

    # Apply the Sobel filter in the y direction
    sobel_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)

    # Combine the Sobel images using bitwise OR
    sobel = cv2.bitwise_or(sobel_x, sobel_y)
    return sobel
def preprocess_image(image):
    # Ensure image is color image with 3 channels
    if len(image.shape) < 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a color image with 3 channels")

    # Enhance contrast


    # Reduce noise

    # Normalize color
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    l_normalized = clahe.apply(l)
    normalized_lab = cv2.merge((l_normalized, a, b))
    normalized_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    return normalized_bgr
def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image
def new_filter_2(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a bilateral filter to remove noise while preserving edges
    gray = cv2.bilateralFilter(gray, 12, 150, 150)

    # Apply local contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Apply adaptive thresholding to segment the blood vessels and little dots
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    # Perform morphological opening to remove small objects
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return opening
def detect_symptoms(img):
    # Load the image
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a median filter to remove noise
    gray = cv2.medianBlur(gray, 5)

    # Apply adaptive thresholding to segment the blood vessels and little dots
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Perform morphological opening to remove small objects
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Detect microaneurysms
    circles = cv2.HoughCircles(opening, cv2.HOUGH_GRADIENT, 1, 200, param1=75, param2=25, minRadius=20, maxRadius=50)

    # Draw circles around the microaneurysms
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    return img



### BEST ###
def preprocess_image(image):
    # Ensure image is color image with 3 channels

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(10,10))
    l_normalized = clahe.apply(l)
    normalized_lab = cv2.merge((l_normalized, a, b))
    normalized_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    return normalized_bgr