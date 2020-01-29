import cv2
from sklearn import svm
import os
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
import argparse
from sklearn.svm import NuSVC
import matplotlib.pyplot as plt
from sklearn import metrics


pos_img_dir = 'C:/Users/poojas/Computer vision/test_pos_person'
neg_img_dir ='C:/Users/poojas/Computer vision/test_neg_person'

clf = joblib.load('person_detection.pkl')

total_pos_samples = 0
total_neg_samples = 0



def filenames():

    f_pos = []
    f_neg = []

    for (dirpath, dirnames, filenames) in os.walk(pos_img_dir):
        f_pos.extend(filenames)
        break

    for (dirpath, dirnames, filenames) in os.walk(neg_img_dir):
        f_neg.extend(filenames)
        break

    print ("Positive Image Samples: " + str(len(f_pos)))
    print ("Negative Image Samples: " + str(len(f_neg)))

    return f_pos, f_neg

def read_images(pos, neg):

    print ("Reading Images")

    array_pos_features = []
    array_neg_features = []
    global total_pos_samples
    global total_neg_samples
    for imgfile in pos:
        img = cv2.imread(os.path.join(pos_img_dir, imgfile))
        #cropped = crop_centre(img)
        #gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(img, orientations=9, pixels_per_cell=(4,4), cells_per_block=(2, 2), block_norm="L2", feature_vector=True)
        array_pos_features.append(features.tolist())
        #print(features.shape)
        total_pos_samples += 1

    for imgfile in neg:
        img = cv2.imread(os.path.join(neg_img_dir, imgfile))
        #cropped = crop_centre(img)
        #gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm="L2", feature_vector=True)
        array_neg_features.append(features.tolist())
        total_neg_samples += 1

    return array_pos_features, array_neg_features

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    plt.savefig('C:/Users/poojas/Downloads/ROC5.png')



pos_img_files, neg_img_files = filenames()

pos_features, neg_features = read_images(pos_img_files, neg_img_files)

pos_result = clf.predict(pos_features)
neg_result = clf.predict(neg_features)
y=[]
l=[]

res=  np.concatenate((pos_result,neg_result), axis=0)
k= pos_result.shape
y= np.ones(k)
l=np.zeros(neg_result.shape)

w=np.concatenate((y,l), axis=0)
fpr, tpr, thresholds = metrics.roc_curve(w, res)



plt.plot(fpr, tpr, color='orange', label='ROC')
#plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve- 4x4 Gamma=0.002')
plt.legend()
#plt.show()
plt.savefig('C:/Users/poojas/Desktop/projectCV/ROC9.png')


auc = metrics.roc_auc_score(w, res)
print('AUC: %.3f' % auc)
plt.ioff()

true_positives = cv2.countNonZero(pos_result)
false_negatives = pos_result.shape[0] - true_positives

false_positives = cv2.countNonZero(neg_result)
true_negatives = neg_result.shape[0] - false_positives

print ("True Positives: " + str(true_positives), "False Positives: " + str(false_positives))
print ("True Negatives: " + str(true_negatives), "False Negatives: " + str(false_negatives))

precision = float(true_positives) / (true_positives + false_positives)
recall = float(true_positives) / (true_positives + false_negatives)

f1 = 2*precision*recall / (precision + recall)

#plot_roc_curve(false_positives, true_positives)

print ("Precision: " + str(precision), "Recall: " + str(recall))
print ("F1 Score: " + str(f1))
