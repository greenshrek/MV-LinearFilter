import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = cv2.imread("cat.png")
greyimg = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)

def showimg(img, greyimg):

    #show the image
    cv2.imshow('image',img)

    #show grescale image
    greyimg = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('grey-image',greyimg)

    #image stays on the screen till any key is pressed
    cv2.waitKey(0)

def sobelFilter(greyimg):

    #horizontal filter
    h_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    #vertical filter
    v_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    fx = cv2.filter2D(greyimg, -1, h_filter)
    cv2.imshow('sobel-filter-horizontal',fx)

    cv2.waitKey(0)

    fy = cv2.filter2D(greyimg, -1, v_filter)
    cv2.imshow('sobel-filter-vertical',fy)

    cv2.waitKey(0)


def gaussianFilterMask(greyimg):
    sigma = 0.5

    x, y = np.meshgrid(np.arange(0,len(greyimg[0])),np.arange(0,len(greyimg)))

    dog_kernel_x = -(x-len(greyimg[0])/2)*np.exp(-((x-len(greyimg[0])/2)**2+(y-len(greyimg)/2)**2)/(2*sigma**2))/(2*np.pi*sigma**4)    
    dog_kernel_y = -(y-len(greyimg)/2)*np.exp(-((x-len(greyimg[0])/2)**2+(y-len(greyimg)/2)**2)/(2*sigma**2))/(2*np.pi*sigma**4)

    dog_x = cv2.filter2D(greyimg, -1, dog_kernel_x)    
    dog_y = cv2.filter2D(greyimg, -1, dog_kernel_y)

    cv2.imshow("dog_x", dog_x/np.max(dog_x))
    cv2.imshow("dog_y", dog_y/np.max(dog_y))


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calFourierTransform(img):
    
    ft = np.fft.fft2(img)

    cv2.imshow("ft",np.fft.fftshift(abs(ft))/np.max(abs(ft))*255)

    inv_ft = np.fft.ifft2(ft)
    cv2.imshow("inv ft",abs(inv_ft)/255)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#task 1
showimg(img, greyimg)

#task 2
sobelFilter(greyimg)

#task3
gaussianFilterMask(greyimg)

#task4
calFourierTransform(greyimg) 