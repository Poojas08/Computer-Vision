import cv2
from sklearn import svm
import os
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.utils import shuffle
import sys
import argparse
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.svm import NuSVC



def windows(img):
    h, w = img.shape
    if h < 96 or w < 160:
        return []

    h = h - 96;
    w = w - 160

    windows = []

    for i in range(0, 10):
        x = random.randint(0, w)
        y = random.randint(0, h)
        windows.append(img[y:y + 96, x:x + 160])

    return windows


f_pos = []
f_neg = []
pos_img_dir ='C:/Users/poojas/Computer vision/pos_person'
mypath_pos = pos_img_dir
count = 0
for (dirpath, dirnames, filenames) in os.walk(mypath_pos):
    f_pos.extend(filenames)
    break
neg_img_dir = 'C:/Users/poojas/Computer vision/neg_person'
mypath_neg = neg_img_dir
for (dirpath, dirnames, filenames) in os.walk(mypath_neg):
    f_neg.extend(filenames)
    break

X = []
Y = []
pos_count = 0

features = []
for img_file in f_pos:
    #print (os.path.join(pos_img_dir, img_file))
    img = cv2.imread(os.path.join(pos_img_dir, img_file))
    #img1=img.resize((480,640))
    # plt.imshow(img)
    #print(img.shape)
    #print (os.path.join(pos_img_dir, img_file))
    # cropped = crop_centre(img)

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img_file)
    features = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm="L2",
                   transform_sqrt=False, feature_vector=True)
    #print(features.shape)
    X.append(features)
    Y.append(1)
    pos_count += 1


print('pos_count', pos_count)

neg_count = 0

for img_file in f_neg:
    #print (os.path.join(neg_img_dir, img_file))
    img = cv2.imread(os.path.join(neg_img_dir, img_file), 0)
    #print(img.shape)
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # samples = np.float32(samples)
    neg_count += 1
   

    features = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm="L2",
                       transform_sqrt=True, feature_vector=True)
    X.append(features)
    Y.append(0)

print('neg_count', neg_count)

# X = np.array(X)
X = np.array(X, np.float32)
# Y = np.array(Y)
Y = np.array(Y, np.float32)

# X= np.float32(X)
# Y = np.float32(Y)
print(Y)
print(Y.shape)

X, Y = shuffle(X, Y, random_state=0)

clf1 = svm.NuSVC(gamma=0.002, max_iter=2000)


clf1.fit(X, Y)
print ("Trained")


joblib.dump(clf1, 'person_detection.pkl')
