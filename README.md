# Floor Tile Planning Using OpenCV and Morphological Transformation
This is project is part of my semester project. So I divided this project into two parts

## **Overview**
This is a Python code that performs image processing on a grayscale image of a room with objects to detect and segment the rooms and objects in the image. The code is divided into two parts. In the first part, the code applies Otsu's thresholding algorithm to convert the input image into a binary image. Then, a connected-component labeling algorithm is applied to identify connected regions in the image. The background component is removed, and the size of each component is calculated. The remaining components are marked in a new binary image. Morphological transformations are applied to the binary image to obtain a better result. In the second part, the code uses the output of Part 1, i.e., the dilated binary image, as input. The code detects the corners of the walls using OpenCV's Corner Harris function and draws lines between them to close the rooms off. The code returns a list of boolean masks for each detected room and the colored version of the input image with each room assigned a random color.

### **Part 1: Object Segmentation**
The first part of the code involves object segmentation using the following steps:

1. Read the input image using the OpenCV package. Convert the image to a binary image using Otsu's thresholding algorithm.
2. Apply a connected-component labeling algorithm to identify connected regions in the image.
Remove the background component, leaving only smaller components.
3. Calculate the size of each component and remove components smaller than a specified minimum size (150 in this case).
4. Apply morphological transformations to the binary image to obtain a better result. Here, dilation is used to thicken the boundaries of the remaining components.
5. Display the resulting image and save it as "Dilation.jpg".


### **Part 2: Room Detection**
The second part of the code uses the output of Part 1, i.e., the dilated binary image, as input. 
The code detects the corners of the walls using OpenCV's Corner Harris function and draws lines between them to close the rooms off. The code performs the following steps:

1. Clean the input image to remove any noise using a thresholding technique.
2. Detect the corners of the walls using OpenCV's Corner Harris function.
3. Draw lines between the corners to close the rooms off.
4. Check if two corners are on the same x or y coordinate, and if the distance between them is less than a specified maximum length (100 in this case). If both conditions are met, a line is drawn between the two corners.
Save the resulting image as "colored_house".
5. Return a list of boolean masks for each detected room, and the colored version of the input image with each room assigned a random color.
Usage


To run the code, you need to have Python and OpenCV installed. Clone the repository and run the code in a Python environment. The input image should be placed in the same directory as the code.

## **Conclusion**
This code is a simple implementation of object segmentation and room detection using Python and OpenCV. It can be used as a starting point for more complex applications, such as object tracking or scene reconstruction.
