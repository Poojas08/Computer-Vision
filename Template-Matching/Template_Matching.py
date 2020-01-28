#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:01:12 2019

@author: poojasharma
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def noisy(image, noise_type, sigma):
    if noise_type == "gauss":
        row, col = image.shape
        mean = 0
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    elif noise_type == "s&p":
        row, col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        row, col = image.shape
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = image + image * gauss
        return noisy

max_corr_val = np.zeros((11,6))
for noise_level in range(11):
    for sigma in range(6):
        kernel_size = sigma*6 + 1;
        template_image = cv2.imread('/Users/poojasharma/Documents/cv/hw2/template.png', cv2.IMREAD_GRAYSCALE)
        main_image = cv2.imread('/Users/poojasharma/Documents/cv/hw2/motherboard-gray.png', cv2.IMREAD_GRAYSCALE)  
        motherboard_with_noise = np.uint8(noisy(main_image, 'gauss', noise_level))                                 # Adding noise level
        motherboard_smoothed = cv2.GaussianBlur(motherboard_with_noise, ksize=(kernel_size, kernel_size), sigmaX=sigma)   #Smoothing 
        template_matching = cv2.matchTemplate(motherboard_smoothed, template_image, cv2.TM_CCORR_NORMED)     #Using template matching function
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_matching, None)                          #finding maximum correlation value
        
        rawindices = np.argsort(-template_matching, axis=None)
        candidateindices = np.unravel_index(rawindices, template_matching.shape)

        matchLoc = max_loc
        cv2.rectangle(motherboard_smoothed, matchLoc,
                      (matchLoc[0] + template_image.shape[0], matchLoc[1] + template_image.shape[1]), (0, 0, 0), 2, 8,
                      0)
        cv2.rectangle(template_matching, matchLoc,
                      (matchLoc[0] + template_image.shape[0], matchLoc[1] + template_image.shape[1]), (0, 0, 0), 2, 8,
                      0)
        max_corr_val[noise_level, sigma] = max_val

print(max_corr_val)

I2= np.uint16(template_matching)

cv2.namedWindow('Template_matching', flags=cv2.WINDOW_NORMAL)
cv2.imshow('Template_matching', template_matching)
cv2.imwrite('/Users/poojasharma/Documents/cv/hw2/template_matching.png', I2)
cv2.namedWindow('Noisy Image', flags=cv2.WINDOW_NORMAL)
cv2.imshow('Noisy Image', motherboard_with_noise)
cv2.imwrite('/Users/poojasharma/Documents/cv/hw2/motherboard_with_noise.png', motherboard_with_noise)
cv2.namedWindow('Smoothed Image', flags=cv2.WINDOW_NORMAL)
cv2.imshow('Smoothed Image', motherboard_smoothed)
cv2.imwrite('/Users/poojasharma/Documents/cv/hw2/motherboard_smoothed.png', motherboard_smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows()
