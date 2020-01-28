import numpy as np
import cv2
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
import os


def trackPoints(movedOutFlag, pts, im, ws):  # % Tracking initial points (pt_x, pt_y) across the image sequence #%

    N = pts.shape[0]
    nim = im.shape[0]
    trackPts = np.zeros([nim, N, 2], dtype=float)

    trackPts[0] = pts[:, 0:2]
    for t in range(nim - 1):
        (movedOutFlag, trackPts[t + 1, :]) = getNextPoints(movedOutFlag, trackPts[t, :], im[t], im[t + 1], ws)
    return (movedOutFlag, trackPts)

def readImages(imgFolder, filename):
        images = []
        for filename in os.listdir(imgFolder):
            img = cv2.imread(os.path.join(imgFolder, filename))
            if img is not None:
                images.append(img)
        return images


def getNextPoints(movedOutFlag, xy, im1, im2, ws):  # % Iterative Lucas-Kanade feature tracking #% x,
    # y : initialized keypoint position in im2 #% x2, y2: tracked keypoint positions in im2 #% ws: patch window size

    xy2 = xy
    x = xy[:, 0]
    y = xy[:, 1]
    x2 = xy2[:, 0]
    y2 = xy2[:, 1]
    (imgRows, imgColumns) = im1.shape
    I = np.float64(im1)
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
    Ixx = np.multiply(Ix, Ix)
    Ixy = np.multiply(Ix, Iy)
    Iyy = np.multiply(Iy, Iy)
    halfSide = np.int32(np.floor(ws / 2))
    sizeRow = xy.shape[0]
    rowInd, colInd = np.meshgrid(range(-halfSide, halfSide + 1, 1), range(-halfSide, halfSide + 1), sparse = False, indexing = 'ij')

    for i in range(sizeRow):
        if movedOutFlag[i] == 0:
            if (x[i] > imgRows - halfSide - 1) or (y[i] > imgColumns - halfSide - 1):
                x2[i] = x[i]
                y2[i] = y[i]
                movedOutFlag[i] = 1
                continue

            if (x[i] < halfSide) or (y[i] < halfSide):
                x2[i] = x[i]
                y2[i] = y[i]
                movedOutFlag[i] = 1
                continue

            xCoor = rowInd + x[i]
            yCoor = colInd + y[i]
            IxPatch = interp2d(Ix, xCoor, yCoor, 'bilinear')
            IyPatch = interp2d(Iy, xCoor, yCoor, 'bilinear')
            imagePatch = interp2d(im1, xCoor, yCoor, 'bilinear')

            for j in range(1, 5):
                xCoor2 = rowInd + x2[i]
                yCoor2 = colInd + y2[i]

                Is = interp2d(im2, xCoor2, yCoor2, 'bilinear');
                It = Is - imagePatch

                if np.isnan(It):
                    continue;
                A = [[sum(sum(IxPatch * IxPatch)), sum(sum(IxPatch * IyPatch))], [sum(sum(IxPatch * IyPatch)), sum(sum(IyPatch * IyPatch))]]
                b = -[[sum(sum(IxPatch * It))], [sum(sum(IyPatch * It))]]
                uv = np.linalg.lstsq(A, b)

                x2[i] = x2[i] + uv[1]
                y2[i] = y2[i] + uv[2]

                if (x2[i] > imgRows - halfSide - 1) or (y2[i] > imgColumns - halfSide - 1):
                    x2[i] = x[i]
                    y2[i] = y[i]
                    movedOutFlag[i] = 1
                    continue

            if (x2[i] < halfSide) or (y2[i] < halfSide):
                x2[i] = x[i]
                y2[i] = y[i]
                movedOutFlag[i] = 1
                continue
    xy2[:, 0] = x2
    xy2[:, 1] = y2
# << << << << YOUR CODE GOES HERE >> >> >> >>
    return (movedOutFlag, xy2)

def getKeypoints(img, threshold=0.485):
    img_cpy = img.copy()
    # Grayscale
    img1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    I = np.float64(img1_gray)
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
    Ixx = np.multiply(Ix, Ix)
    Ixy = np.multiply(Ix, Iy)
    Iyy = np.multiply(Iy, Iy)

    Sxx = np.float64(cv2.GaussianBlur(Ixx, (7, 7), 1))
    Sxy = np.float64(cv2.GaussianBlur(Ixy, (7, 7), 1))
    Syy = np.float64(cv2.GaussianBlur(Iyy, (7, 7), 1))
    height, width = I.shape
    harris = (Sxx * Syy - np.square(Sxy)) - (0.06 * np.square(Sxx + Syy))
    cv2.normalize(harris, harris, 0, 1, cv2.NORM_MINMAX)
    # harris=Non_max_suppression(harris,5)
    harris_max = np.amax(harris)
    print(harris_max)
    threshoLd = 0.9 * harris_max / 2

    # find all points above threshold
    loc = np.where(harris >= threshold)
    # loop though the points
    for pt in zip(*loc[::-1]):
        # draw filled circle on each point
        cv2.circle(img_cpy, pt, 3, (0, 255, 0), -1)
    return img_cpy


folder = "hotelImages"

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

tau = 0.06
imagesInFolder = 50
images = readImages(folder, "hotel.seq")
currImage = images[0]
results = getKeypoints(currImage, tau)
movedOutFlag = np.zeros(results.shape[0])
patchSize = 15
hw = np.floor(patchSize / 2)# flags for starting points patchSize = 15   # Tracking patchSize by patchSize patches
(movedOutFlag, trackXY) = trackPoints(movedOutFlag, results, images, patchSize)

image = cv2.cvtColor(currImage, cv2.COLOR_GRAY2BGR)

for result in (results):
    cv2.circle(image, (np.uint32(result[0]), np.uint32(result[1])), 4, GREEN)
for ptIndex in range(100):
    if(movedOutFlag[ptIndex] == 0):
        for loc in (trackXY[:, ptIndex, :]):
            cv2.circle(image, (np.uint32(loc[0]), np.uint32(loc[1])), 2, RED)
cv2.waitKey(0)


