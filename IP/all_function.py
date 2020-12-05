import cv2
import numpy as np
import math
from scipy import ndimage
import functools

def all(input, output):

    # adaptive_threshold.py
    img = cv2.imread(input,0)
    img = cv2.medianBlur(img,5)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    # erosion-dilation.py
    kernel = np.ones((3,3),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1)
    img = cv2.erode(img,kernel,iterations = 1)

    img = cv2.medianBlur(img,5)

    # main_axis2.py
    h, w = img.shape
    mat = np.argwhere(img != 255)
    mat[:, [0, 1]] = mat[:, [1, 0]]
    mat = np.array(mat).astype(np.float32)
    m, e = cv2.PCACompute(mat, mean = np.array([])) # อ่าน PCA
    center = tuple(m[0])
    endpoint1 = tuple(m[0] + e[0]*100)
    endpoint2 = tuple(m[0] + e[1]*50)
    delta0 = endpoint1[0]-center[0]
    delta1 = endpoint1[1]-center[1]
    angle = math.atan2(delta0, delta1)
    angle = angle*180/math.pi
    print(angle)
    inv = cv2.bitwise_not(img)
    rotated = ndimage.rotate(inv, -angle+90)
    inv = cv2.bitwise_not(rotated)
    img = inv

    # auto_crop2.py
    points = np.argwhere(img==0)
    points = np.fliplr(points)
    x, y, w, h = cv2.boundingRect(points)
    crop = img[y:y+h, x:x+w]
    img = crop

    # resize2.py
    height = 100
    width = int(img.shape[1] * height / img.shape[0])
    dim = (width, height)
    resize = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = resize

    # add_white.py
    desired_size = 500
    old_size = img.shape[:2]
    old_size_int = functools.reduce(lambda sub, ele: sub * 10 + ele, old_size)
    top = bottom = left = 0
    right = desired_size - old_size[1]
    color = [255, 255, 255]
    white = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    img = white

    cv2.imwrite(output,img)

for index in range(1, 11):
    infile = '%s%s.jpg' % ('Wachiragorn\\', str(index))
    outfile = '%sall%s.jpg' % ('Wachiragorn\\', str(index))
    all(infile, outfile)