import cv2
import numpy as np
import matplotlib.pyplot as plt
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
 
def alignImages(im1, im2):

    orb = cv2.ORB_create(MAX_FEATURES)
    image1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    image2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)


    keypoints1, descriptors1 = orb.detectAndCompute(image1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2Gray, None)
    #keypoints3, descriptors3 = orb.detectAndCompute(im3Gray, None)

    # Match features.
    feature_matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    feature_matches = feature_matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    feature_matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches`
    numGoodMatches = int(len(feature_matches) * GOOD_MATCH_PERCENT)
    matches = feature_matches[:numGoodMatches]
    print(cv2.DMatch().distance)
    # Draw top matches
    imageMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imageMatches)
    cv2.imshow("matches.jpg", imageMatches)
    plt.imshow(imageMatches)
    cv2.waitKey(0)
    # Extract location of good matches
    point1 = np.zeros((len(imageMatches), 2), dtype=np.float32)
    point2 = np.zeros((len(imageMatches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        point2[i, :] = keypoints1[match.queryIdx].pt
        point1[i, :] = keypoints2[match.trainIdx].pt
    displacement=np.abs(point2-point1)
    
    x_val, x_cnt = np.unique(displacement[:,0],return_counts=True)
    x_disp = int(x_val[np.argmax(x_cnt)])

    y_val, y_cnt = np.unique(displacement[:,1],return_counts=True)
    y_disp =int( y_val[np.argmax(y_cnt)])
    print(y_disp)
    print(x_disp)
    # Find homography
    h, mask = cv2.findHomography(point1, point2, cv2.RANSAC, 5)
    
    print(h)

    mat_out = h.dot(np.array([im2.shape[1], im2.shape[0], 1]))
    print(mat_out)

    width =int( im2.shape[1] + np.abs(x_disp))

    height =int( max(im1.shape[0], mat_out[1]))
    print(height, width)

    # Use homography
    im_Reg2 = cv2.warpPerspective(im2, h, (width, height)) 
    im_Reg2[0:im1.shape[0], 0:im1.shape[1]] = im1        

    # Read reference image

    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
    return im2Reg, h
 
Img_1 = cv2.imread('/Users/poojasharma/desktop/hobbit0.png', cv2.IMREAD_COLOR)
Img_2 =cv2.imread('/Users/poojasharma/Downloads/hobbit1.png', cv2.IMREAD_COLOR)
Img_3 = cv2.imread('/Users/poojasharma/Downloads/hobbit2.png', cv2.IMREAD_COLOR)

first_combined, h = alignImages(Img_1,Img_2)
cv2.imshow('Result', first_combined)
cv2.imshow("original_image_stitched_crop.jpg",(first_combined))
outFilename = "aligned_image.jpg"
print("Saving aligned image : ", outFilename)  
cv2.imwrite(outFilename, trim(first_combined))
combined_out1=cv2.imread('aligned_image.jpg',cv2.IMREAD_COLOR)
second_combined , h1 = alignImages(first_combined,Img_3)

cv2.imshow('Combined',(second_combined))
cv2.waitKey(0)
cv2.destroyAllWindows()  
# Write aligned image to disk.
outFile1 = "final_image.png"
print("Saving aligned image : ", outFile1)

cv2.imwrite(outFile1,second_combined)
# Print estimated homography
print("Estimated homography : \n",  h1)

