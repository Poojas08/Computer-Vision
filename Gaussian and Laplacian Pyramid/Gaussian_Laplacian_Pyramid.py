"""
Created on Tue Sep 24 22:27:03 2019

@author: poojasharma
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/poojasharma/Documents/cv/pepper.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original image", img)

Kx= cv2.getGaussianKernel(ksize=11,sigma=1.5)
Ky=np.transpose(Kx)
kernel= Kx*Ky
ksize= 11

height = len(img)
print(height)
width = len(img [0])
print(width)
print(img.shape)
# create an output image
res = np.zeros((height, width, 1), dtype= "uint8")

temp = img
images=[temp]

dst=np.zeros((height, width, 1), dtype= "uint8")
Lap=np.zeros((height, width, 1), dtype= "uint8")
expandi=np.zeros((height, width, 1), dtype= "uint8")
result=np.zeros((height, width, 1), dtype= "uint8")
#GAUSSIAN PYRAMID

for i in range(5):
    plt.subplot(1, 5, i + 1)
    dst= cv2.GaussianBlur(temp,(11,11),1.5)
    dst= dst[::2,1::2]    
    temp=dst
    images.append(temp)
   
    cv2.imshow("str(i)", dst)
    f = np.fft.fft2(temp)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum_G = 20*np.log(np.abs(fshift))
   
   
    plt.subplot(121),plt.imshow(temp, cmap = 'gray')
    plt.title('Input Image_G'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum_G, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.imshow(magnitude_spectrum_G)
    #plt.imsave('/Users/poojasharma/Documents/cv/'+ str("Gaussian") + str(i)+'.jpg', magnitude_spectrum_G)
   
    cv2.imshow("str(i)", dst)
    cv2.imwrite('/Users/poojasharma/Documents/cv/'+ str("gaussian") + str(i)+'.jpg', dst)
   
   
   
#LAPLACIAN PYRAMID
for i in range(5):
    plt.subplot(1, 5, i + 1)
    dst= cv2.GaussianBlur(temp,(11,11),1.5)
    dst= dst[::2,1::2]
    Lap=dst
    h,w=Lap.shape
    print(Lap.shape)
    expandi= cv2.resize(Lap,(h*2, w*2))
    print(expandi.shape)
    result = cv2.subtract(temp,expandi)
    temp=dst
    images.append(temp)
   
    f1 = np.fft.fft2(result)
    fshift = np.fft.fftshift(f1)
    magnitude_spectrum_L = 20*np.log(np.abs(fshift))
   
   
    plt.subplot(121),plt.imshow(result, cmap = 'gray')
    plt.title('Input Image_L'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum_L, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.imsave('/Users/poojasharma/Documents/cv/'+ str("Laplacian") + str(i)+'.jpg', magnitude_spectrum_L)
    plt.show()
    plt.imshow(result)
    cv2.imshow("str(i)", result)
   
    cv2.imwrite('/Users/poojasharma/Documents/cv/'+ str("Laplacian") + str(i)+'.jpg', result)
