import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdf2image
import json
from copy import copy

POPPLER_PATH = r"C:\Users\CF6P\Downloads\Release-23.01.0-0\poppler-23.01.0\Library\bin"

def PDF_to_images(path, POPPLER=POPPLER_PATH):
    images = pdf2image.convert_from_path(path, poppler_path=POPPLER)
    return [np.array(image) for image in images]

def preprocessed_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return thresh

def _points_filter(points):
    """
    Get the endpoint along each axis
    """
    points[points < 0] = 0
    xpoints = sorted(points, key=lambda x:x[0])
    ypoints = sorted(points, key=lambda x:x[1])
    tpl_x0, tpl_x1 = xpoints[::len(xpoints)-1]
    tpl_y0, tpl_y1 = ypoints[::len(ypoints)-1]
    return tpl_y0[1], tpl_y1[1], tpl_x0[0], tpl_x1[0]


def get_rectangle(processed_image, kernel_size=(3,3), interations = 2):
    """
    Crop and rotate the image thanks to contour detection
    """
    if interations == 2:
        format = "other"
    else:
        format = "table"
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilate = cv2.dilate(~processed_image, kernel, iterations=interations)
    # plt.imshow(dilate)
    # plt.show()
    contours,_ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No contour found : Crop is impossible, processed image is return.")
        return []
    y, x = processed_image.shape[:2]
    rectangles = [cv2.minAreaRect(contour) for contour in contours]
    rectangles = [rect for rect in rectangles if rect[1][0]*rect[1][1]>x*y/12]

    if len(rectangles)==1:
        return format, rectangles[0]
        
    if len(rectangles)>1:
        return get_rectangle(processed_image, kernel_size=(5,5), interations = interations+2)


def crop_and_rotate(processed_image, rect):
    if len(rect)==0 : 
        return processed_image
    box = np.intp(cv2.boxPoints(rect))    
    # Rotate image
    angle = rect[2]
    if 45<=angle<=90: 
        angle = angle-90 # Angle correction
    rows, cols = processed_image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1) # Rotation matrix
    img_rot = cv2.warpAffine(processed_image,M,(cols,rows))
    # rotate bounding box, then crop
    points = np.intp(cv2.transform(np.array([box]), M))[0] # Points of angles of the box after rotatio,
    y0, y1, x0, x1 = _points_filter(points) # get the biggest crop
    cropped_image = img_rot[y0:y1, x0:x1]
    # plt.imshow(cropped_image)
    # plt.axis('off')
    # plt.show()
    return cropped_image

    
if __name__ == "__main__":
    
    print("start")
    path = r"C:\Users\CF6P\Desktop\cv_text\Data\scan3.pdf"
    images = PDF_to_images(path)
    res_dict_per_image = {}
    for i, image in enumerate(images,1):
        print(f"\nImage {i} is starting")
        processed_image = preprocessed_image(image)
  
        
##########################################################""
        
    def get_lines(image):
        edges = cv2.Canny(image, 50, 255)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,300,minLineLength=100,maxLineGap=50)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        print(len(lines))
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(image,(x1,y1),(x2,y2),(255,0,0),3)
        plt.imshow(image)
        plt.show()
        return image
    
    def process_contours(contours, processed_image):
        y, x = processed_image.shape[:2]
        rectangles = [cv2.minAreaRect(contour) for contour in contours]
        rectangles = [rect for rect in rectangles if rect[1][0]*rect[1][1]>x*y/12]
        print(len(rectangles))
        if len(rectangles)==1:
            return "other", rectangles[0]
        
        if len(rectangles)>1: # Cret a new rectangle from found
            sorted_rect = sorted(rectangles, key = lambda x: x[0][0]) # Sorted by the height of the center
            print(sorted_rect)
            mean = lambda x,y,i,j: (x[i][j] + y[i][j])/2
            diff = lambda x,y,i,j: abs((x[i][j] - y[i][j]))
            w = max(rectangles, key = lambda x: x[1][0])[1][0]
            x, y = mean(sorted_rect[0], sorted_rect[-1], 0, 0), mean(sorted_rect[0], sorted_rect[-1], 0, 1)
            dx, dy = diff(sorted_rect[0], sorted_rect[-1], 0, 0), diff(sorted_rect[0], sorted_rect[-1], 0, 1)  
            h = dx + sorted_rect[0][1][1]/2 + sorted_rect[-1][1][1]/2
            theta = 90-np.arctan(dx/dy)
            rectangle = ((x,y), (w,h), theta)
            return "table", rectangle