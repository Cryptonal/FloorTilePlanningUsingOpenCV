#Import packages
import os
import cv2
import numpy as np
import sys

# Name of the directory containing the object detection module we're using
sys.path.append("..")

IMAGE_NAME = 'roomArc.PNG'
#Remove Small Items
im_gray = cv2.imread(IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh = 185
im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
#find all your connected components
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im_bw, connectivity=8)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 150

#your answer image
img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255

cv2.imshow('room detector:Step 1', img2)
#MorphologicalTransform
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(img2, kernel, iterations=1)
##erosion = cv2.erode(img2, kernel, iterations=6)

#cv2.imshow("img2", img2)
cv2.imshow("Dilation: Step 2", dilation)
cv2.imwrite("Dilation.jpg", dilation)
##cv2.imshow("Erosion", erosion)
##Part 1 completed----------------------------------------------------------------------------///

##Part 2:

##2nd Part (finding rooms and detecting corners)


def find_rooms(img, noise_removal_threshold=25, corners_threshold=0.1,
               room_closing_max_length=100, gap_in_wall_threshold=500):
    """

    :param img: grey scale image of rooms, already eroded and doors removed etc.
    :param noise_removal_threshold: Minimal area of blobs to be kept.
    :param corners_threshold: Threshold to allow corners. Higher removes more of the house.
    :param room_closing_max_length: Maximum line length to add to close off open doors.
    :param gap_in_wall_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
    :return: rooms: list of numpy arrays containing boolean masks for each detected room
             colored_house: A colored version of the input image, where each room has a random color.
    """
    
    assert 0 <= corners_threshold <= 1
    # Remove noise left from door removal

    img[img < 128] = 0
    img[img > 128] = 255
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        ##Currently, the area is in pixel measurements, since input is an pixeled image. But if we know the area of each pixel of image we can convert it to square meters unit.
        print("Area of contour " + str(i) + ": " + str(area))
        tile_area = 10 * 10  # Assuming each tile is 10x10 square units.
        num_tiles = (area // tile_area)
        if(area % tile_area == 0):
            print("No of tiles required are:" + str(num_tiles) + "\n")
        else:
            print("No of tiles required are:" + str(num_tiles + 1) + "\n")

        if area > noise_removal_threshold:
            cv2.fillPoly(mask, [contour], 255)

    img = ~mask

    # Detect corners (you can play with the parameters here)
    dst = cv2.cornerHarris(img ,2,3,0.04)
    dst = cv2.dilate(dst,None)
    corners = dst > corners_threshold * dst.max()

    # Draw lines to close the rooms off by adding a line between corners on the same x or y coordinate
    # This gets some false positives.
    # You could try to disallow drawing through other existing lines for example.
    for y,row in enumerate(corners):
        x_same_y = np.argwhere(row)
        for x1, x2 in zip(x_same_y[:-1], x_same_y[1:]):
            if x2[0] - x1[0] < room_closing_max_length:
                if x1 and y and x2:
                    color = 0
                    cv2.line(img, (x1[0], y), (x2[0], y), color, 1)

                else:
                    x1 = x2 = y = 0
                    color = 0
                    cv2.line(img, (x1, y), (x2, y), color, 1)

    for x,col in enumerate(corners.T):
        y_same_x = np.argwhere(col)
        for y1, y2 in zip(y_same_x[:-1], y_same_x[1:]):
            if y2[0] - y1[0] < room_closing_max_length:
                if x and y1 and y2:
                    color = 0
                    cv2.line(img, (x, y1[0]), (x, y2[0]), color, 1)

                else:
                    x = y1 = y2 = 0
                    color = 0
                    cv2.line(img, (x, y1), (x, y2), color, 1)


    # Mark the outside of the house as black
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [biggest_contour], 255)
    img[mask == 0] = 0

    # Find the connected components in the house
    ret, labels = cv2.connectedComponents(img)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    unique = np.unique(labels)
    rooms = []
    for label in unique:
        component = labels == label
        if img[component].sum() == 0 or np.count_nonzero(component) < gap_in_wall_threshold:
            color = 0
        else:
            rooms.append(component)
            color = np.random.randint(0, 255, size=3)
        img[component] = color

    return rooms, img

#Read gray image
img = cv2.imread("DilationImage.PNG", 0)
rooms, colored_house = find_rooms(img.copy())
cv2.imshow('Step 3: Result', colored_house)

###2nd Part Completed----------------------------------------------------------------------//


# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()