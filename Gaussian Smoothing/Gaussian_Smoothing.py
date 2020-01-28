"""
Created on Mon Sep 23 22:26:27 2019

@author: poojasharma
"""
#Computer Vision HW1 Ques1 Pooja Sharma
import numpy as np
import cv2
import matplotlib.pyplot as plt


print("OPEN CV VERSION", cv2.__version__)
# Load an image in grayscale
img   = cv2.imread('/Users/poojasharma/Documents/cv/lighthouse.bmp', cv2.IMREAD_GRAYSCALE)
height = len  (img  )
print(height)
width = len  (img  [0])
print(width)
print(img.shape)
# create an output image
res = np.zeros((height, width, 1), dtype= "uint8")
# apply a kernel
Kx= cv2.getGaussianKernel(ksize=11,sigma=1.5)
Ky=np.transpose(Kx)
kernel= Kx*Ky                        #creating a 2-D Kernel
ksize= 11


for r in range(height-ksize):       #2D Gaussian Blurring
    for c in range(width-ksize):
        pixel = 0  
        for y in range(ksize):
            for x in range(ksize):
                pixel += kernel[x,y]* img  [r+y, c+x]
                res[r, c] = min(255, int(abs(pixel)));
               
               
               
               
diff=np.zeros((height, width, 1), dtype= "uint8")
result = np.zeros((height, width, 1), dtype= "uint8")
resultf = np.zeros((height, width, 1), dtype= "uint8")
kernel2= cv2.getGaussianKernel(ksize=11,sigma=1.5)         #1D kernel
kernel3=np.transpose(kernel2)                                               #1D Kernel


for r in range(height-ksize):
    for c in range(width-ksize):
        pixel = 0  
        for y in range(1):
            for x in range(ksize):
                pixel += kernel2[x,y]* img  [r+y, c+x]
                result[r, c] = min(255, int(abs(pixel)));
               
               
resultf = np.zeros((height, width, 1), dtype= "uint8")
 
for r in range(height-ksize):
    for c in range(width-ksize):
        pixel = 0  
        for x in range(1):
            for y in range(ksize):
                pixel += kernel3[x,y]*result[r+y, c+x]
                resultf[r, c] = min(255, int(abs(pixel)));  
               
               
for r in range(height):                   #Difference between 2D Gaussian Burring and 1D GAussian blurring
    for c in range(width):
        diff[r,c]=res[r,c]-resultf[r,c]
       
print('Mean:',np.mean(diff))         #Calculate Mean
print('Variance:',np.var(diff))      #Calculate Variance
print('Median:',np.median(diff))     #Calculate Median
o_img=cv2.vconcat([res,resultf])
o_img1=cv2.vconcat([diff,img])
output_image=cv2.hconcat([o_img,o_img1])      
cv2.namedWindow('OUTPUT', flags=cv2.WINDOW_NORMAL)
'''cv2.resizeWindow('INPUT', (int(width/2), int(height/2)))
cv2.namedWindow('RESULT', flags=cv2.WINDOW_NORMAL)'''
cv2.imwrite('/Users/poojasharma/Documents/cv/'+ str("Result2") +'.jpg', output_image)
cv2.imshow('RESULT',output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

