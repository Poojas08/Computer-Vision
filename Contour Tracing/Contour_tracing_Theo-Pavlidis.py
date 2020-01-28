import array
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image

pathName = "/Users/poojasharma/Documents/cv/hw3/"

MAXCONTOUR = 5000
doLogging = False


def showImage(img, name):
    cv2.imshow(name, img)
    return


################################
# change this for your own file structure
def saveImage(img, name):
    cv2.imwrite(pathName + name + ".png", img)
    return


def Pavlidis(img, start):
    x, y = img.shape
    col, row = start
    ystt = row
    xstt = col
    print(ystt, xstt)

    contour = []

    i = 0
    Direction = []
    Direction = ['Direction_UP', 'Direction_right', 'Direction_down', 'Direction_left']
    px = 0
    py = 0

    direction = Direction[0]
    print(direction)
    num_contour = 0

    while (num_contour < 5000):
        if px == xstt and py == ystt:
            break
        rotation = 0
        # print(direction)
        # print(num_contour)

        for rotation in range(4):
            if direction == 'Direction_UP':
                Ly, Lx = row - 1, col - 1
                Cy, Cx = row - 1, col
                Ry, Rx = row - 1, col + 1

            if direction == 'Direction_right':
                Ly, Lx = row - 1, col + 1
                Cy, Cx = row, col + 1
                Ry, Rx = row + 1, col + 1

            if (direction == 'Direction_down'):
                Ly, Lx = row + 1, col + 1
                Cy, Cx = row + 1, col
                Ry, Rx = row + 1, col - 1

            if (direction == 'Direction_left'):
                Ly, Lx = row + 1, col - 1
                Cy, Cx = row, col - 1
                Ry, Rx = row - 1, col - 1

            if img[Ly, Lx] != 0:
                col, row = Lx, Ly
                contour.append([Lx, Ly])
                # print([Lx, Ly])
                i = (i + 3) % 4
                direction = Direction[i]
                # print(direction)
                break

            elif img[Cy, Cx] != 0:
                col, row = Cx, Cy
                contour.append([Cx, Cy])
                # print(Cx, Cy)
                # print(direction)
                break

            elif img[Ry, Rx] != 0:
                col, row = Rx, Ry
                contour.append([Rx, Ry])
                # print([Rx, Ry])
                # print(direction)
                break

            # rotation+=1
            i = i + 1
            direction = Direction[i % 4]


        px = col
        py = row
        # print(py, px)
        num_contour += 1
    print(num_contour)
    print(contour)
    return contour



def showContour(ctr, img, name):
    contourImage = img
    length = len(ctr)
    # print(length)
    for count in range(length):
        contourImage[ctr[count, 1], ctr[count, 0]] = 255
        cv2.line(contourImage, (ctr[count, 0], ctr[count, 1]), \
                 (ctr[(count + 1) % length, 0], ctr[(count + 1) % length, 1]), (128, 128, 128), 1)
        showImage(contourImage, name)
        saveImage(contourImage, name)
        plt.imshow(contourImage)
    return


#
def GaussArea(pts):
    #   <<<<< YOUR CODE HERE >>>>>
    length = len(pts)
    area = 0
    for i in range(length):
        area += (pts[i, 1] * pts[(i + 1)% length, 0] - pts[i, 0] * pts[(i + 1)%length, 1])
    return np.abs(area/2);


def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]

    return z


def onePassDCE(contour):
    min_contour = 1000
    relevance_measure = np.array([])
    # print("Initial contour length", len(contour))
    for i in range(len(contour)):
        angle1 = (math.atan2((contour[i, 1] - contour[i - 1, 1]), (contour[i, 0] - contour[i - 1, 0])))
        angle2 = (math.atan2((contour[(i + 1)%len(contour), 1] - contour[i, 1]), (contour[(i + 1)%len(contour), 0] - contour[i, 0])))
        turn_angle = np.abs(angle1 - angle2)

        length1 = math.sqrt(((contour[i, 1]) - contour[i - 1, 1]) ** 2 + (contour[i, 0] - contour[i - 1, 0]) ** 2)
        length2 = math.sqrt(((contour[(i + 1)%len(contour), 1]) - contour[i, 1]) ** 2 + (contour[(i + 1)%len(contour), 0] - contour[i, 0]) ** 2)

        relevance_measure = np.append(relevance_measure, np.float((turn_angle * length1 * length2) / (length1 + length2)))

        # min_value = min(min_contour, relevance_measure[i])
        # trimmed_contour = sort_list(contour, relevance_measure)

        # del trimmed_contour[-1]
    # print("Relevance Measure Length", len(relevance_measure))
    # contour = contour[np.argsort(relevance_measure)]
    trimmed_contour = np.delete(contour, np.argmin(relevance_measure), 0)
    # print("Contour Length : ", len(contour))
    # trimmed_contour = contour[1:]
    # print("Trimmed Contour", len(trimmed_contour))
    return trimmed_contour


inputImage = cv2.imread('/Users/poojasharma/Downloads/VAoutline.png', cv2.IMREAD_GRAYSCALE)
thresh = 70
cv2.imshow('image', inputImage)

binary = cv2.threshold(inputImage, thresh, 255, cv2.THRESH_BINARY)[1]
print('start')
plt.imshow(binary)
(height, width) = binary.shape

print(height, width)
ystt = np.uint8(height / 2)  # look midway up the image
for xstt in range(width):  # from the left
    if binary[ystt, xstt] > 0:
        break

contour = np.array(Pavlidis(binary, [xstt, ystt]))
# mask = np.zeros(inputImage.shape, np.uint8)
# cv2.drawContours(mask, [contour], -1, 255, -1)
showContour(contour, np.zeros(inputImage.shape, np.uint8), "CONTOUR")

area = GaussArea(contour)

for step in range(6):
    numLoops = math.floor(contour.shape[0]/2)
    for idx in range(numLoops):
        contour = onePassDCE(contour)
       # print("Length", len(contour))
    showContour(contour, np.zeros_like(inputImage), "STEP"+str(step))
    print('Loop:',step,'numLoops:',numLoops, 'contour_shape:',contour.shape, 'GaussArea(contour):',GaussArea(contour))
cv2.waitKey(0)
