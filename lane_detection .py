import cv2
import numpy as np
import matplotlib
from tkinter import *
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
matplotlib.use('TkAgg')
def make_coordinates(image,line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit,axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in  lines:
            x1 ,y1 ,x2 ,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255, 0 ,0), 10)
    return line_image
def region_of_intrest(image):
    height = image.shape[0]
    polygons = np.array([[(0, height) , (1250, height) , (600,150)]])      #200 , 1100, (550,250)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


image = cv2.imread('WhatsApp Image 2021-07-13 at 17.15.47.jpeg')
lane_image = np.copy(image)
canny_image  = canny(lane_image)
cropped_image = region_of_intrest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 20 , maxLineGap = 10)
averaged_lines = average_slope_intercept(lane_image,lines)
line_image = display_lines(lane_image, averaged_lines)
display_image = cv2.addWeighted(lane_image, 0.8 ,line_image,1,1)
cv2.imshow("result" , display_image)
cv2.waitKey(0)

#cap =cv2.VideoCapture("test2.mp4")
#while(cap.isOpened()):
#    _, frame = cap.read()
#    canny_image = canny(frame)
#    cropped_image = region_of_intrest(canny_image)
#    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=20, maxLineGap=10)
#    averaged_lines = average_slope_intercept(frame, lines)
#    line_image = display_lines(frame, averaged_lines)
#    display_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#    cv2.imshow("result", display_image)
#    if cv2.waitKey(1) == ord('c'):
#        break
#cap.release()
cv2.destroyAllWindows()