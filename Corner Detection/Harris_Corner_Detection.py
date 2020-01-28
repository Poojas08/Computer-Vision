import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage,misc
import os
img = cv2.imread("Users/poojasharma/Documents/cv/hw2/hotel/hotel.seq0.png')
# create copy
imagesInFolder=50
offset=2
window_size=5
kernel = np.ones((5,5), np.uint8)
imgFolder = "/Users/poojasharma/Documents/cv/hw2/hotel/"
erosion_size=(2*1) + 1
def readImages(imgFolder,filename):
    images = []
    for filename in os.listdir(imgFolder):
        img = cv2.imread(os.path.join(imgFolder,filename))
        if img is not None:
            images.append(img)
    return images
def Non_max_suppression(img,window_size):
    height=img.shape[0]
    width=img.shape[1]
    for i in range(height-window_size+1):
        for j in range(width-window_size+1):
           window=img[i:i+window_size,j:j+window_size]
           if np.sum(window)==0:
              max_local=0
           else:
              max_local=np.argmax(window)
           label=np.argmax(window)
           window.flat[label]=max_local
    return img          
   
def harris(img, threshold=0.485):   #THRESHOLD= 90% of maximum value of corner strength

    img_cpy = img.copy()
    # Grayscale
    img1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    I = np.float64(img1_gray)    
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)    
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)    
    Ixx = np.multiply(Ix, Ix)    
    Ixy = np.multiply(Ix, Iy)    
    Iyy = np.multiply(Iy, Iy)
   
    Sxx=np.float64(cv2.GaussianBlur(Ixx,(7,7),1))
    Sxy=np.float64(cv2.GaussianBlur(Ixy,(7,7),1))
    Syy=np.float64(cv2.GaussianBlur(Iyy,(7,7),1))
    height,width= I.shape

    harris = (Sxx*Syy - np.square(Sxy)) - (0.06*np.square(Sxx + Syy))
    print(harris.shape)
    

    cv2.normalize(harris, harris, 0, 1, cv2.NORM_MINMAX)
    harris_max = np.amax(harris)
    print(harris_max)
    Threshold = 0.9*harris_max/2

    #print(threshold)
    # loop though the points
    #harris=ndimage.rank_filter(harris,rank=9,size=5)
    loc = np.where(harris >= threshold)
    corner_array = np.array([])
    for pt in zip(*loc[::-1]):
        # draw filled circle on each point
        corner_array = np.append(corner_array, pt)
        cv2.circle(img_cpy, pt, 3, (0, 255, 0), -1)
       
        # draw filled circle on each point
    print(corner_array.shape)
    
    return img_cpy
images=readImages(imgFolder,"hotel.seq")
corners = harris(images[0])
# display images
plt.figure(figsize=(10, 10))
plt.subplot(121), plt.imshow(img)
plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(corners)
plt.title("Harris Corners"), plt.xticks([]), plt.yticks([])
plt.show()
cv2.imwrite("q1.png", cv2.cvtColor(corners, cv2.COLOR_BGR2RGB))

cv2.destroyAllWindows()
