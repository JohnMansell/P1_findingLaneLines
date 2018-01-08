"""-------------------------------------------------
        Udacity
        Self Driving Car
        Project 1 -- Finding Lane Lines

        - John Mansell -

------------------------------------------------"""

'''===================================
        Import
======================================'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

'''===================================
        Helper Functions
======================================'''

'''-------------------
    Show
----------------------'''
def show(img):
    '''
    Display the image to the user
        - determine if the image is gray or RGB
        - display the image with the correct encoding
    '''
    if len(img.shape) > 2:
        plt.imshow(img)
        plt.show()
    else:
        plt.imshow(img, cmap='gray')
        plt.show()

'''-------------------
    GrayScale
----------------------'''
def grayscale(img):
    """Applies the Grayscale transform
        call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


'''-------------------
    Region of Interest
----------------------'''
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

'''-------------------
    Draw Lines
----------------------'''
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# Python 3 has support for cool math symbols.

'''-------------------
    Weighted Image
----------------------'''
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

'''===================================
        Frame Class
======================================'''
class Frame(object):
    """-------------------
        Init
    -------------------"""
    def __init__(self, image):
        """Return a new Image Object"""
        self.color = image
        self.size = (self.color.shape[1], self.color.shape[0])  # xMax, yMax
        self.xMax = self.color.shape[1]
        self.yMax = self.color.shape[0]

    '''-------------------
            Show
    ----------------------'''
    def show(self):
        plt.imshow(self.color)

    '''-------------------
            Gray Scale
    ----------------------'''
    def grayScale(self):
        grayImage = np.copy(self.color)
        return cv2.cvtColor(grayImage, cv2.COLOR_RGB2GRAY)

    '''-------------------
            Get Vertices
    ----------------------'''
    def getVertices(self):
        imshape = self.color.shape
        top_left = (450, imshape[0] / 2)
        bottom_left = (0, imshape[0])
        top_right = (450, imshape[0] / 2)
        bottom_right = (imshape[1], imshape[0])

        vertices = np.array([[(bottom_left), top_left, (top_right), (bottom_right)]], dtype=np.int32)
        return vertices

    '''-------------------
            Masked
    ----------------------'''
    def masked(self):
        vertices = self.getVertices()
        grayPic = self.grayScale()
        return region_of_interest(grayPic, vertices)


'''------------------------------
    Import Images Into Array
---------------------------------'''
import glob
from PIL import Image

fileList = glob.glob('test_images/*.jpg')
images = []

for file in fileList:
    image = plt.imread(file)
    images.append(image)


frame1 = Frame(images[0])

'''------------------------------
    Convert Images to GrayScale
---------------------------------'''
grayPic = frame1.grayScale()


'''------------------------------
        Gausian Blur
---------------------------------'''
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.bilateralFilter(img, kernel_size, 75, 75)

kernel_size = 9

blurPic = gaussian_blur(grayPic, kernel_size)
show(blurPic)

'''------------------------------
        Canny Edge Detection
---------------------------------'''
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

# Define Thresholds
low_threshold = 50
high_threshold = 150

# Create new picture from Canny Edge detection
edges_1 = canny(blurPic, low_threshold, high_threshold)
show(edges_1)
'''------------------------------
        Isolate Region of
        Interest
---------------------------------'''
vertices = frame1.getVertices()
maskedPic = region_of_interest(edges_1, vertices)
show(maskedPic)

'''------------------------------
        Hough Transform
---------------------------------'''


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Define the Hough transform parameters

rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 1   # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 #minimum number of pixels making up a line
max_line_gap = 2    # maximum gap in pixels between connectable line segments

hough_image = hough_lines(maskedPic, rho, theta, threshold, min_line_length, max_line_gap)

show(hough_image)

'''------------------------------
        Plot the Hough Lines
        Onto the Original Image
---------------------------------'''
# Draw hough lines on the edge image
lines_edges = weighted_img(hough_image, frame1.color)
show(lines_edges)

'''------------------------------
        Extrapolate Lines
---------------------------------'''


def extrapolate_lines(line_array, top_of_image):
    # Array to store X and Y values for left and right lane lines
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    length_right = []
    length_left = []

    # Origin is in upper left corner
    top = top_of_image
    for line in line_array:
        for x1, y1, x2, y2 in line:
            # Move origin to bottom left corner
            y1 = top - y1
            y2 = top - y2

            # y = mx + b
            m = (y1 - y2) / (x1 - x2)
            b = y1 - (m * x1)

            # seperate left and right lane lines
            if m < -0.1 and b > 400:
                right_x.append(x1)
                right_x.append(x2)
                right_y.append(y1)
                right_y.append(y2)

            elif m > 0.1 and b < 70:
                left_x.append(x1)
                left_x.append(x2)
                left_y.append(y1)
                left_y.append(y2)

    # Find Best Fit Lines
    rightSlope, rightIntercept = np.polyfit(right_x, right_y, 1)
    leftSlope, leftIntercept = np.polyfit(left_x, left_y, 1)

    right_lane_line = (rightSlope, rightIntercept)
    left_lane_line = (leftSlope, leftIntercept)

    return (right_lane_line, left_lane_line)


lines = cv2.HoughLinesP(maskedPic, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
right_line, left_line = extrapolate_lines(lines, frame1.color.shape[0])

'''------------------------------
        Draw Extrapolated Lines
        onto a blank Image
---------------------------------'''


def draw_lane_lines(img, line1, line2):
    """
    line1, line2 = (m, b) for y = m(x) + b
    img = original image

    """
    # Convert (y = mx + b) into (x1, y1, x2, y2)
    # x = -b/m

    rightSlope = line1[0]
    rightIntercept = line1[1]

    leftSlope = line2[0]
    leftIntercept = line2[1]

    # Use more readable variables
    bottom = img.shape[0]
    b = rightIntercept
    m = rightSlope
    x = int(-b / m)

    # Define the x-intercept of the right lane line
    rightLaneBottom = (x, bottom)

    # Left
    b = leftIntercept
    m = leftSlope
    x = int(-b / m)
    leftLaneBottom = (x, bottom)

    # solve for intersection of 2 lines
    a1 = rightSlope
    a2 = leftSlope
    b1 = rightIntercept
    b2 = leftIntercept

    # Solve for intersection of 2 lines
    a = np.array([[a1, -1], [a2, -1]])
    b = np.array([b1, b2])
    p = np.linalg.solve(a, b)

    # Convert intersection p into img coordinates
    intersection = (int(-(p[0])), int(540 + p[1]))

    # Create blank image
    projection = np.copy(img) * 0
    GREEN = [0, 100, 0]

    # Draw Lines on Blank Image
    cv2.line(projection, rightLaneBottom, intersection, GREEN, thickness=10)
    cv2.line(projection, leftLaneBottom, intersection, GREEN, thickness=10)
    return projection


# Call Draw Lane Lines function
projection = draw_lane_lines(frame1.color, right_line, left_line)
show(projection)

'''------------------------------
        Merge the Images
---------------------------------'''

final_img = weighted_img(projection, frame1.color)
show(final_img)
'''------------------------------
        Save the Processed final
        Image
---------------------------------'''
# Convert from RGB to BGR for saving the image
colorConverted = np.copy(final_img)
cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR, colorConverted)
cv2.imwrite('test_images/output1.png', colorConverted)

'''------------------------------
        Repeat for the rest of
        the 'test images' directory
---------------------------------'''
# Define Parameters:

# Gausian Blur Kernel Size
kernel_size = 5

# Canny Thresholds
low_threshold = 50
high_threshold = 150

# Hough Transform
rho = 1
theta = np.pi/180
threshold = 2
min_line_length = 10
max_line_gap = 2

i = 0

for image in images:
    # Import imgage, convert to grayscale, and isolate region of interest
    frame1 = Frame(image)
    grayPic = frame1.grayScale()
    blurPic = gaussian_blur(grayPic, kernel_size)
    edges_1 = canny(blurPic, low_threshold, high_threshold)
    vertices = frame1.getVertices()
    maskedPic = region_of_interest(edges_1, vertices)

    # Find and average the Lane lines over 10 Images
    lines = cv2.HoughLinesP(maskedPic, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    right_line, left_line = extrapolate_lines(lines, image.shape[0])

    # Draw the lanes on the img
    projection = draw_lane_lines(frame1.color, right_line, left_line)
    final_img = weighted_img(projection, frame1.color)

    # Convert to BGR colorspace for saving
    colorConverted = np.copy(final_img)
    cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR, colorConverted)

    # Save to file
    file_name = 'test_images/output' + str(i) + '.jpg'
    cv2.imwrite(file_name, colorConverted)
    i += 1
'''------------------------------
        Test on Videos
---------------------------------'''
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


'''------------------------------
        Average Lane Lines
---------------------------------'''
# Define an array for storing the lane lines of each frame
smooth_right = []
smooth_left = []


def average_lines(newLine, laneLines, frames_to_smooth=10):
    """
    Take in the L+R lane lines from each frame.
    If less than 10 lines are saved, add the new lines to the array.
    Drop lines more than 10 frames old
    Average the lines in the array.
    Return the new averaged lines

    """

    # Append new lines until array has 10 frames
    if len(laneLines) < frames_to_smooth:
        laneLines.append(newLine)
    else:
        # Shift each frame down by 1. Drop the oldest.
        # Add newest frame to the end of the array.
        for n in range(1, frames_to_smooth):
            laneLines[n - 1] = laneLines[n]

        laneLines[(frames_to_smooth) - 1] = newLine

    # Take the average of all the lines in the array.
    average_line = np.average(laneLines, axis=0)

    return average_line


'''------------------------------
        Define the Process for
        each frame of the video
---------------------------------'''


def process_image(image):
    """
    Process a single frame from a video stream.
    Identify lane lines and plot the line in green on the original image.

    """
    # Import imgage, convert to grayscale, and isolate region of interest
    frame1 = Frame(image)
    grayPic = frame1.grayScale()
    blurPic = gaussian_blur(grayPic, kernel_size)
    edges_1 = canny(blurPic, low_threshold, high_threshold)
    vertices = frame1.getVertices()
    maskedPic = region_of_interest(edges_1, vertices)

    # Find and average the Lane lines over 10 Images
    lines = cv2.HoughLinesP(maskedPic, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    right_line, left_line = extrapolate_lines(lines, image.shape[0])
    ave_right = average_lines(right_line, smooth_right)
    ave_left = average_lines(left_line, smooth_left)

    # Draw the lanes on the img and return to video
    projection = draw_lane_lines(frame1.color, ave_right, ave_left)
    final_img = weighted_img(projection, frame1.color)

    return final_img


'''------------------------------
        Import Video
---------------------------------'''
# Define the output file locations
white_output = ("test_videos_output/solidWhiteRight.mp4")
yellow_output = ('test_videos_output/solidYellowLeft.mp4')

# Define the input file locations
clip1 = VideoFileClip('test_videos/solidWhiteRight.mp4')
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')

# Process each frame in the video
white_clip = clip1.fl_image(process_image)
yellow_clip = clip2.fl_image(process_image)

# Write the video to file
white_clip.write_videofile(white_output, audio=False)
yellow_clip.write_videofile(yellow_output, audio=False)







































