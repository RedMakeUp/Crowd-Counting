# This file is used to generate ground truth(GT) density map based on Guassian Kernel with fixed size and bandwidth.
# See paper [Crowd counting with crowd attention convolutional neural network. Wang et al. 2020. Neurocomputing]

import cv2# For iamge processing
import os# For file input/output
import matplotlib.pyplot as plt# For showing figure
import numpy as np# For n-dimention array
from scipy.io import loadmat# For reading .mat file


def GetIdFromImageName(imageName: str) -> str:
    '''
    Parse image id from its name.
    For example: 
        INPUT: IMG_1.jpg
        OUTPUT: 1
    '''
    index1 = imageName.rfind("_")
    index2 = imageName.rfind(".")
    return imageName[index1+1 : index2]

def MatlabStyleGauss2D(size,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    u = cv2.getGaussianKernel(size, sigma)
    kernel = np.outer(u, u)
    return kernel


# Const variables
PATH_TO_DATASET_IMAGE = "G:/Dataset/ShanghaiTech/part_A_final/train_data/images"# Directory of the images
PATH_TO_DATASET_LOCATION = "G:/Dataset/ShanghaiTech/part_A_final/train_data/ground_truth"# Direfctory of the annotations
PREFIX_LOCATION_NAME = "GT_IMG_"# E.g. GT_IMG_1, GT_IMG_2, GT_IMG_3...
# Variables
densityMaps = []# List of all density map and each density map is a dictionary with structure {"id", "map"}
kernel = MatlabStyleGauss2D(15, 4)# Fixed guassian kernel with size 15*15 and sigma(or bandwidth) 4


# Read images and initialize corresponding density map
with os.scandir(PATH_TO_DATASET_IMAGE) as entries:
    for entry in entries:
        # Relative path to absolute path
        fileName = PATH_TO_DATASET_IMAGE + "/" + entry.name
        # Read the image. Numpy array of [row(height) * column(width) * channels(B,G,R,...)]
        image = cv2.imread(fileName)
        # Change the order of channels from BRG to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Create a 2-d array with the same size as the image but with only one channel, and initialize all elements with 0
        densityMap = np.zeros((image.shape[0], image.shape[1]))
        # Get id of the image
        imageId = GetIdFromImageName(fileName)
        densityMaps.append({
            "id": imageId,
            "map": densityMap
        })

# Read annotation data(.mat) and add "headLocations" field to densitymaps
for entry in densityMaps:
    annotationFilePath = PATH_TO_DATASET_LOCATION + "/" + PREFIX_LOCATION_NAME + entry["id"] + ".mat"
    annotation = loadmat(annotationFilePath)
    annotation = annotation["image_info"][:][0][0][0][0][0]
    entry["headLocations"] = annotation

# Now in densityMaps, we have all we need to generate density map
# TODO



